from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pandas as pd
import io
import requests
import os
import re
import contextlib

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

df_cache = None
user_context = []  # Memoria de preguntas/respuestas

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    global df_cache
    try:
        contents = await file.read()
        df_raw = pd.read_csv(io.BytesIO(contents))
        df_raw = df_raw.loc[:, df_raw.columns.str.strip() != ""]
        df_cache = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')]

        print("CSV cargado:")
        print(df_cache.head())
        return {"message": "Archivo CSV cargado correctamente"}
    except Exception as e:
        return {"error": f"Error al procesar el archivo CSV: {str(e)}"}

@app.post("/preguntar/")
async def preguntar(pregunta: str = Form(...)):
    global df_cache, user_context
    if df_cache is None:
        return {"error": "No se ha subido un archivo CSV aún"}

    try:
        context = df_cache.to_csv(index=False)
        columnas = df_cache.columns.tolist()

        if len(context) > 12000:
            context = df_cache.head(50).to_csv(index=False)

        # Construir mensaje inicial + contexto
        system_prompt = {
            "role": "system",
            "content": (
                "Eres un experto en análisis de datos con Python y pandas.\n"
                f"El DataFrame se llama 'df' y contiene estas columnas: {columnas}\n"
                "Responde únicamente con código Python, usando solo 'df'."
            )
        }

        # Añadir la nueva pregunta al historial
        user_context.append({"role": "user", "content": pregunta})
        # Limitar contexto a los últimos 6 mensajes (3 rondas)
        memory = user_context[-6:]

        messages = [system_prompt] + memory

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
            "Authorization": f"Bearer {GROQ_API_KEY}",
        }

        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        if response.status_code != 200:
            return {"error": f"Error del modelo: {response.text}"}

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        # Añadir respuesta al historial
        user_context.append({"role": "assistant", "content": answer})

        match = re.search(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
        code_str = match.group(1).strip() if match else answer.strip()

        print("Código generado por el modelo:\n", code_str)

        local_vars = {'df': df_cache.copy()}
        output = io.StringIO()

        try:
            with contextlib.redirect_stdout(output):
                exec(code_str, {}, local_vars)

            result = local_vars.get('result', output.getvalue().strip())

        except Exception as e:
            return {
                "codigo_python": code_str,
                "error": f"Error al ejecutar el código: {str(e)}"
            }

        # Función para sanear valores no serializables
        def sanitize(val):
            try:
                if pd.isna(val) or val in [float('inf'), float('-inf')]:
                    return None
                return val
            except:
                return val

        if isinstance(result, pd.DataFrame):
            clean_df = result.copy().applymap(sanitize)
            result_json = {
                "tipo": "DataFrame",
                "columnas": list(clean_df.columns),
                "filas": clean_df.to_dict(orient="records")
            }

        elif isinstance(result, (list, dict)):
            def clean_container(container):
                if isinstance(container, dict):
                    return {k: sanitize(v) for k, v in container.items()}
                elif isinstance(container, list):
                    return [sanitize(item) for item in container]
                return sanitize(container)

            result_json = {
                "tipo": type(result).__name__,
                "valor": clean_container(result)
            }

        elif isinstance(result, str):
            try:
                df_like = pd.read_fwf(io.StringIO(result))
                if not df_like.empty:
                    clean_df = df_like.copy().applymap(sanitize)
                    result_json = {
                        "tipo": "DataFrame",
                        "columnas": list(clean_df.columns),
                        "filas": clean_df.to_dict(orient="records")
                    }
                else:
                    raise ValueError("No es tabla válida")
            except Exception:
                result_json = {
                    "tipo": "str",
                    "valor": sanitize(result)
                }

        else:
            result_json = {
                "tipo": "texto",
                "salida": sanitize(str(result))
            }

        print("Resultado generado:", result_json)

        return {
            "codigo_python": code_str,
            "resultado": result_json
        }

    except Exception as e:
        return {
            "codigo_python": code_str if 'code_str' in locals() else "",
            "error": f"Error inesperado: {str(e)}"
        }
