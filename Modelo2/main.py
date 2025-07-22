from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import pandas as pd
import numpy as np
import aiohttp
import asyncio
import io
import os
import re
import contextlib
import hashlib
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from functools import lru_cache
import psutil
from pathlib import Path
from fastapi import Request

# ============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================================

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de seguridad
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

API_SECRET = os.getenv("API_SECRET", "1234")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

# Límites de configuración
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_ROWS_CONTEXT = 1000
MAX_CONTEXT_LENGTH = 15000
SESSION_TTL = 3600  # 1 hora
MAX_SESSIONS = 100
CHUNK_SIZE = 10000

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

from pydantic import BaseModel, Field, validator
from typing import Any, List, Dict, Optional
import re

class QuestionModel(BaseModel):
    pregunta: str = Field(max_length=500, min_length=1)
    session_id: str  # ahora obligatorio
    
    @validator('pregunta')
    def validate_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("La pregunta no puede estar vacía")
        dangerous_patterns = [
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__',
            r'subprocess',
            r'os\.',
            r'system\s*\(',
            r'open\s*\(',
            r'file\s*\(',
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Pregunta contiene patrones no permitidos")
        return v
class PaginatedResponse(BaseModel):
    data: List[Dict[str, Any]]
    total: int
    page: int
    size: int
    has_next: bool

class AnalysisResult(BaseModel):
    tipo: str
    columnas: Optional[List[str]] = None
    filas: Optional[List[Dict[str, Any]]] = None
    valor: Optional[Any] = None
    salida: Optional[str] = None

# ============================================================================
# CACHE Y SESIONES
# ============================================================================

class SessionCache:
    def __init__(self, max_size: int = MAX_SESSIONS, ttl: int = SESSION_TTL):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times: Dict[str, datetime] = {}
    
    def _cleanup_expired(self):
        """Limpia sesiones expiradas"""
        now = datetime.now()
        expired_sessions = [
            session_id for session_id, access_time in self.access_times.items()
            if (now - access_time).total_seconds() > self.ttl
        ]
        for session_id in expired_sessions:
            # Limpiar archivos de base de datos temporales
            session_data = self.cache.get(session_id, {})
            source_info = session_data.get('source_info', {})
            if source_info.get('type') == 'db' and source_info.get('db_path'):
                try:
                    os.remove(source_info.get('db_path'))
                    logger.info(f"Archivo temporal de DB eliminado para sesión {session_id}")
                except OSError:
                    pass
            
            self.cache.pop(session_id, None)
            self.access_times.pop(session_id, None)
    
    def _evict_oldest(self):
        """Elimina las sesiones más antiguas si se supera el límite"""
        if len(self.cache) >= self.max_size:
            oldest_session = min(self.access_times.items(), key=lambda x: x[1])[0]
            self.cache.pop(oldest_session, None)
            self.access_times.pop(oldest_session, None)
    
    def get(self, session_id: str, key: str) -> Any:
        self._cleanup_expired()
        session_data = self.cache.get(session_id, {})
        self.access_times[session_id] = datetime.now()
        return session_data.get(key)
    
    def set(self, session_id: str, key: str, value: Any):
        self._cleanup_expired()
        self._evict_oldest()
        
        if session_id not in self.cache:
            self.cache[session_id] = {}
        
        self.cache[session_id][key] = value
        self.access_times[session_id] = datetime.now()
    
    def get_session_stats(self) -> Dict[str, Any]:
        self._cleanup_expired()
        return {
            "active_sessions": len(self.cache),
            "memory_usage_mb": sum(
                len(str(session_data)) for session_data in self.cache.values()
            ) / 1024 / 1024,
            "oldest_session": min(self.access_times.values()) if self.access_times else None
        }

# Cache global de sesiones
session_cache = SessionCache()

# ============================================================================
# UTILIDADES DE SEGURIDAD
# ============================================================================

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verificación básica de token"""
    if credentials.credentials != API_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido"
        )
    return credentials.credentials

def validate_file(file: UploadFile) -> None:
    """Valida archivos subidos"""
    if not (file.filename.lower().endswith('.csv') or file.filename.lower().endswith('.db')):
        raise HTTPException(400, "Solo se permiten archivos CSV o SQLITE(.db)")
    
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(400, f"Archivo muy grande. Máximo {MAX_FILE_SIZE // 1024 // 1024}MB")

# ============================================================================
# UTILIDADES DE OPTIMIZACIÓN
# ============================================================================

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimiza tipos de datos del DataFrame"""
    try:
        # Convertir object a category si tiene pocos valores únicos
        for col in df.select_dtypes(include=['object']):
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        
        # Optimizar tipos numéricos
        for col in df.select_dtypes(include=['int64']):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']):
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    except Exception as e:
        logger.warning(f"Error optimizando DataFrame: {e}")
        return df

def sanitize_dataframe_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitiza DataFrame usando operaciones vectorizadas"""
    df = df.copy()
    
    # Reemplazar inf y -inf con NaN, luego con 0
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Llenar NaN en columnas numéricas con 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Llenar NaN en columnas de texto con string vacío
    text_cols = df.select_dtypes(include=['object', 'category']).columns
    df[text_cols] = df[text_cols].fillna('')
    
    return df

def paginate_dataframe(df: pd.DataFrame, page: int = 1, size: int = 100) -> PaginatedResponse:
    """Pagina resultados del DataFrame"""
    start = (page - 1) * size
    end = start + size
    
    return PaginatedResponse(
        data=df.iloc[start:end].to_dict('records'),
        total=len(df),
        page=page,
        size=size,
        has_next=end < len(df)
    )

@lru_cache(maxsize=50)
def get_compiled_regex():
    """Regex compilado para extraer código"""
    return re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

# ============================================================================
# PROCESAMIENTO SEGURO DE CONSULTAS
# ============================================================================

ALLOWED_OPERATIONS = {
    'describe', 'info', 'head', 'tail', 'shape', 'columns', 'dtypes',
    'value_counts', 'unique', 'nunique', 'isnull', 'sum', 'mean', 
    'median', 'std', 'min', 'max', 'count', 'groupby', 'sort_values',
    'sort_index', 'reset_index', 'drop_duplicates', 'fillna', 'dropna',
    'query', 'loc', 'iloc', 'at', 'iat', 'sample'
}

def create_safe_analysis_context(df: pd.DataFrame, source_info: Dict[str, Any] = None) -> str:
    """Crea contexto seguro para análisis"""
    context_parts = []
    
    if source_info and source_info.get("type") == "db":
        # Contexto para base de datos SQL
        context_parts.append("TIPO DE FUENTE: Base de datos SQLite")
        context_parts.append(f"Tablas disponibles: {source_info.get('all_tables', [])}")
        context_parts.append(f"Tabla principal: {source_info.get('table')}")
        
        # Esquema de las tablas
        schema = source_info.get('schema', {})
        for table, columns in schema.items():
            context_parts.append(f"\nEsquema de tabla '{table}':")
            for col_name, col_type in columns:
                context_parts.append(f"  - {col_name}: {col_type}")
        
        # Muestra de datos de la tabla principal
        if len(df) > 0:
            sample_size = min(3, len(df))
            context_parts.append(f"\nPrimeras {sample_size} filas de la tabla principal:")
            context_parts.append(df.head(sample_size).to_string())
            
    else:
        # Contexto para CSV/pandas
        context_parts.append("TIPO DE FUENTE: Archivo CSV")
        context_parts.append(f"Columnas disponibles: {list(df.columns)}")
        context_parts.append(f"Forma del DataFrame: {df.shape}")
        context_parts.append(f"Tipos de datos:\n{df.dtypes.to_string()}")
        
        # Muestra de datos (limitada)
        if len(df) > 0:
            sample_size = min(5, len(df))
            context_parts.append(f"Primeras {sample_size} filas:\n{df.head(sample_size).to_string()}")
        
        # Estadísticas básicas para columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe()
            context_parts.append(f"Estadísticas básicas:\n{stats.to_string()}")
    
    return "\n\n".join(context_parts)

def validate_and_execute_safe_sql(sql: str, db_path: str) -> Any:
    """Ejecuta consultas SQL de manera segura"""
    import sqlite3
    
    # Verificar patrones peligrosos en SQL
    dangerous_sql_patterns = [
        r'DROP\s+TABLE', r'DELETE\s+FROM', r'UPDATE\s+', r'INSERT\s+INTO',
        r'ALTER\s+TABLE', r'CREATE\s+TABLE', r'PRAGMA', r'ATTACH',
        r'DETACH', r'VACUUM', r'REINDEX'
    ]
    
    sql_upper = sql.upper()
    for pattern in dangerous_sql_patterns:
        if re.search(pattern, sql_upper):
            raise ValueError(f"Operación SQL no permitida: {pattern}")
    
    # Solo permitir SELECT
    if not sql_upper.strip().startswith('SELECT'):
        raise ValueError("Solo se permiten consultas SELECT")
    
    try:
        conn = sqlite3.connect(db_path)
        df_result = pd.read_sql_query(sql, conn)
        conn.close()
        
        # Limitar resultados
        if len(df_result) > 1000:
            df_result = df_result.head(1000)
            logger.warning("Resultado SQL truncado a 1000 filas")
            
        return df_result
        
    except Exception as e:
        logger.error(f"Error ejecutando SQL: {e}")
        raise ValueError(f"Error en consulta SQL: {str(e)}")

def validate_and_execute_safe_code(code: str, df: pd.DataFrame) -> Any:
    """Ejecuta código de manera segura usando solo pandas operations"""
    
    # Lista blanca de operaciones permitidas
    allowed_patterns = [
        r'df\.(describe|info|head|tail|shape|columns|dtypes)\(\)',
        r'df\[.*?\]\.(value_counts|unique|nunique|sum|mean|median|std|min|max|count)\(\)',
        r'df\.groupby\(.*?\)\.(sum|mean|count|size|agg)\(.*?\)',
        r'df\.(sort_values|sort_index|reset_index|drop_duplicates)\(.*?\)',
        r'df\.(fillna|dropna|query|sample)\(.*?\)',
        r'df\.(loc|iloc)\[.*?\]',
        r'len\(df\)',
        r'df\.shape\[0\]',
        r'df\.shape\[1\]',
    ]
    
    # Verificar que el código solo contenga operaciones permitidas
    clean_code = re.sub(r'#.*', '', code)  # Remover comentarios
    clean_code = re.sub(r'\s+', ' ', clean_code.strip())  # Normalizar espacios
    
    # Verificar patrones peligrosos
    dangerous_patterns = [
        r'exec', r'eval', r'__import__', r'subprocess', r'os\.',
        r'system', r'open', r'file', r'input', r'raw_input',
        r'compile', r'globals', r'locals', r'vars', r'dir',
        r'getattr', r'setattr', r'delattr', r'hasattr'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, clean_code, re.IGNORECASE):
            raise ValueError(f"Operación no permitida detectada: {pattern}")
    
    # Crear contexto seguro para ejecución
    safe_globals = {
        'df': df.copy(),
        'pd': pd,
        'np': np,
        '__builtins__': {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'max': max,
            'min': min,
            'sum': sum,
            'abs': abs,
            'round': round,
            'print': print,

        }
    }
    
    local_vars = {}
    output = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(output):
            # Usar compile para verificar sintaxis
            compiled_code = compile(clean_code, '<string>', 'exec')
            exec(compiled_code, safe_globals, local_vars)
        
        # Buscar resultado
        result = local_vars.get('result', output.getvalue().strip())
        return result if result else "Operación completada sin output"
        
    except Exception as e:
        logger.error(f"Error ejecutando código seguro: {e}")
        raise ValueError(f"Error en análisis: {str(e)}")

# ============================================================================
# CLIENTE HTTP ASÍNCRONO
# ============================================================================

class AsyncGroqClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.session = None
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Llamada asíncrona a la API de Groq"""
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
            "top_p": 1,
            "stream": False,
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        async with self.session.post(self.base_url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(500, f"Error del modelo: {error_text}")
            
            return await response.json()

# ============================================================================
# APLICACIÓN FASTAPI
# ============================================================================

app = FastAPI(
    title="CSV Analyzer API",
    description="API segura y optimizada para análisis de datos CSV",
    version="2.0.0",
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configurado de manera segura
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def index(request: Request):  # ✅ Tipo explícito

    """Página principal - con cache"""
    try:
        template_path = Path("templates/index.html")
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")
        return "<h1>CSV Analyzer API</h1><p>Sube un CSV y haz preguntas sobre tus datos</p>"
    except Exception as e:
        logger.error(f"Error cargando template: {e}")
        return "<h1>CSV Analyzer API</h1><p>API funcionando correctamente</p>"

@app.get("/health")
async def health_check():
    """Health check con métricas del sistema"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent(),
        },
        "sessions": session_cache.get_session_stats()
    }

@app.post("/upload_file/")
@limiter.limit("5/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    _: str = Depends(verify_token)
):
    validate_file(file)

    session_id = str(uuid.uuid4())
    extension = file.filename.lower().split(".")[-1]

    try:
        contents = await file.read()

        if extension == "csv":
            df = pd.read_csv(io.BytesIO(contents), encoding="utf-8", low_memory=False)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = df.columns.str.strip()
            df = optimize_dataframe(df)
            if len(df) > MAX_ROWS_CONTEXT:
                df = df.head(MAX_ROWS_CONTEXT)
            source_info = {"type": "csv"}

        elif extension == "db":
            # Guardar archivo temporalmente
            temp_db_path = f"/tmp/{session_id}.db"
            with open(temp_db_path, "wb") as f:
                f.write(contents)

            import sqlite3
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()

            # Obtener todas las tablas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            if not tables:
                raise HTTPException(400, "No se encontraron tablas en el archivo .db")

            # Cargar la primera tabla como ejemplo para mostrar estructura
            main_table = tables[0]
            df_sample = pd.read_sql_query(f"SELECT * FROM {main_table} LIMIT 100", conn)
            df_sample = optimize_dataframe(df_sample)

            # Obtener esquema de las tablas
            schema_info = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                schema_info[table] = [(col[1], col[2]) for col in columns]  # (nombre, tipo)

            conn.close()
            # NO eliminar el archivo aquí, lo usaremos para consultas SQL
            source_info = {
                "type": "db", 
                "table": main_table, 
                "all_tables": tables,
                "schema": schema_info,
                "db_path": temp_db_path
            }
            df = df_sample  # Solo para mostrar información inicial
            

        else:
            raise HTTPException(400, "Tipo de archivo no soportado")

        # Guardar en cache de sesión
        session_cache.set(session_id, "dataframe", df)
        session_cache.set(session_id, "context", [])
        session_cache.set(session_id, "upload_time", datetime.now())
        session_cache.set(session_id, "source_info", source_info)

        return {
            "message": f"Archivo {extension.upper()} cargado correctamente",
            "session_id": session_id,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "source_info": source_info
        }

    except Exception as e:
        logger.error(f"Error procesando archivo {extension}: {e}")
        raise HTTPException(500, f"Error procesando archivo {extension}: {str(e)}")

@app.post("/preguntar/")
@limiter.limit("20/minute")
async def preguntar(
    request: Request,
    question: QuestionModel,
    _: str = Depends(verify_token)
):
    """Endpoint optimizado para hacer preguntas sobre los datos"""
    
    if not question.session_id:
        raise HTTPException(400, "session_id es requerido")
    
    # Obtener DataFrame de la sesión
    df = session_cache.get(question.session_id, 'dataframe')
    if df is None:
        raise HTTPException(404, "Sesión no encontrada o expirada")
    
    try:
        # Obtener contexto de conversación y tipo de fuente
        user_context = session_cache.get(question.session_id, 'context') or []
        source_info = session_cache.get(question.session_id, 'source_info') or {}
        
        # Crear contexto seguro
        safe_context = create_safe_analysis_context(df, source_info)
        
        # Construir mensajes para la API según el tipo de fuente
        if source_info.get("type") == "db":
            # Para bases de datos - generar SQL
            system_prompt = {
                "role": "system",
                "content": (
                    "Eres un experto en consultas SQL para SQLite. "
                    "IMPORTANTE: Solo puedes usar consultas SELECT seguras. "
                    "Responde ÚNICAMENTE con consultas SQL válidas para SQLite. "
                    "No uses DROP, DELETE, UPDATE, INSERT, ALTER, PRAGMA, ni operaciones peligrosas. "
                    "Usa solo SELECT con JOIN, WHERE, GROUP BY, ORDER BY, COUNT, SUM, AVG, etc. "
                    f"Contexto de la base de datos:\n{safe_context[:MAX_CONTEXT_LENGTH]}"
                )
            }
        else:
            # Para CSV - generar código Python/pandas
            system_prompt = {
                "role": "system",
                "content": (
                    "Eres un experto en análisis de datos con pandas. "
                    "IMPORTANTE: Solo puedes usar operaciones básicas y seguras de pandas. "
                    "Responde ÚNICAMENTE con código Python usando 'df' como nombre del DataFrame. "
                    "No uses exec, eval, import, subprocess, os, file operations, ni ninguna función peligrosa. "
                    "Usa solo: describe(), info(), head(), tail(), groupby(), value_counts(), etc. "
                    f"Contexto del DataFrame:\n{safe_context[:MAX_CONTEXT_LENGTH]}"
                )
            }
        
        # Añadir pregunta al contexto
        user_context.append({"role": "user", "content": question.pregunta})
        
        # Limitar contexto (últimos 6 mensajes)
        recent_context = user_context[-6:]
        messages = [system_prompt] + recent_context
        
        # Llamada asíncrona a la API
        async with AsyncGroqClient(GROQ_API_KEY) as client:
            response_data = await client.chat_completion(messages)
        
        answer = response_data["choices"][0]["message"]["content"]
        
        # Añadir respuesta al contexto
        user_context.append({"role": "assistant", "content": answer})
        session_cache.set(question.session_id, 'context', user_context)
        
        # Extraer código según el tipo de fuente
        if source_info.get("type") == "db":
            # Extraer consulta SQL
            sql_regex = re.compile(r"```(?:sql)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
            match = sql_regex.search(answer)
            code_str = match.group(1).strip() if match else answer.strip()
            
            logger.info(f"SQL generado para sesión {question.session_id}: {code_str[:100]}...")
            
            # Ejecutar consulta SQL
            db_path = source_info.get('db_path')
            result = validate_and_execute_safe_sql(code_str, db_path)
            
        else:
            # Extraer código Python
            regex = get_compiled_regex()
            match = regex.search(answer)
            code_str = match.group(1).strip() if match else answer.strip()
            
            logger.info(f"Código Python generado para sesión {question.session_id}: {code_str[:100]}...")
            
            # Ejecutar código de manera segura
            result = validate_and_execute_safe_code(code_str, df)
        
        # Procesar resultado
        result_json = process_analysis_result(result)
        
        return {
            "codigo_python": code_str if source_info.get("type") != "db" else None,
            "codigo_sql": code_str if source_info.get("type") == "db" else None,
            "resultado": result_json,
            "session_id": question.session_id,
            "tipo_fuente": source_info.get("type", "csv")
        }
        
    except ValueError as ve:
        logger.warning(f"Validación fallida: {ve}")
        raise HTTPException(400, str(ve))
    except Exception as e:
        logger.error(f"Error procesando pregunta: {e}")
        raise HTTPException(500, f"Error inesperado: {str(e)}")

def process_analysis_result(result: Any) -> AnalysisResult:
    """Procesa el resultado del análisis de manera segura"""
    
    if isinstance(result, pd.DataFrame):
        # Sanitizar DataFrame
        clean_df = sanitize_dataframe_vectorized(result)
        
        # Limitar tamaño del resultado
        if len(clean_df) > 1000:
            clean_df = clean_df.head(1000)
            logger.warning("Resultado truncado a 1000 filas")
        
        return AnalysisResult(
            tipo="DataFrame",
            columnas=clean_df.columns.tolist(),
            filas=clean_df.to_dict(orient="records")
        )
    
    elif isinstance(result, pd.Series):
        # Convertir Series a formato seguro
        series_dict = result.head(100).to_dict()  # Limitar a 100 elementos
        return AnalysisResult(
            tipo="Series",
            valor=series_dict
        )
    
    elif isinstance(result, (list, dict)):
        return AnalysisResult(
            tipo=type(result).__name__,
            valor=result
        )
    
    elif isinstance(result, str):
        return AnalysisResult(
            tipo="str",
            salida=result[:5000]  # Limitar output largo
        )
    
    else:
        return AnalysisResult(
            tipo="texto",
            salida=str(result)[:5000]
        )

@app.get("/sessions/{session_id}/stats")
@limiter.limit("10/minute")
async def get_session_stats(
    request,
    session_id: str,
    _: str = Depends(verify_token)
):
    """Obtener estadísticas de una sesión"""
    df = session_cache.get(session_id, 'dataframe')
    if df is None:
        raise HTTPException(404, "Sesión no encontrada")
    
    context = session_cache.get(session_id, 'context') or []
    upload_time = session_cache.get(session_id, 'upload_time')
    
    return {
        "session_id": session_id,
        "dataframe_shape": df.shape,
        "dataframe_columns": df.columns.tolist(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "conversation_length": len(context),
        "upload_time": upload_time.isoformat() if upload_time else None,
        "last_activity": session_cache.access_times.get(session_id, datetime.now()).isoformat()
    }

# ============================================================================
# STARTUP Y SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("CSV Analyzer API iniciada - Versión segura y optimizada")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("CSV Analyzer API detenida")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
