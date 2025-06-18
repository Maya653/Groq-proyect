from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pandas as pd
import io
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
df_cache = None

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    global df_cache
    try:
        contents = await file.read()
        df_cache = pd.read_csv(io.BytesIO(contents))
        return {"message": "Archivo CSV cargado correctamente"}
    except Exception as e:
        return {"error": f"Error al procesar el archivo CSV: {str(e)}"}

@app.post("/preguntar/")
async def preguntar(pregunta: str = Form(...)):
    global df_cache
    if df_cache is None:
        return {"error": "No se ha subido un archivo CSV aún"}

    try:
        # Convertir CSV a texto (sin index)
        context = df_cache.to_csv(index=False)

        # Limitar contexto si es demasiado grande
        if len(context) > 12000:
            context = df_cache.head(50).to_csv(index=False)

        prompt = f"""
Aquí tienes el contenido de un archivo CSV:

{context}

Con base en este contenido, responde la siguiente pregunta como un experto en análisis de datos:

{pregunta}

Si tu respuesta incluye datos en forma de tabla, preséntalos en formato Markdown.
"""

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1,
            "max_tokens": 1024,  # corregido aquí
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

        return {"respuesta": answer}

    except Exception as e:
        return {"error": f"Error en la petición: {str(e)}"}
