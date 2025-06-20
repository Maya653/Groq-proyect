from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pandas as pd
import io
import requests
import os
import re
import sys
import contextlib

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

df_cache = None

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
        df_cache = pd.read_csv(io.BytesIO(contents))
        print("CSV cargado:")
        print(df_cache.head())
        return {"message": "Archivo CSV cargado correctamente"}
    except Exception as e:
        return {"error": f"Error al procesar el archivo CSV: {str(e)}"}

@app.post("/preguntar/")
async def preguntar(pregunta: str = Form(...)):
    global df_cache
    if df_cache is None:
        return {"error": "No se ha subido un archivo CSV aún"}

    try:
        context = df_cache.to_csv(index=False)
        if len(context) > 12000:
            context = df_cache.head(50).to_csv(index=False)

        prompt = f"""
Eres un experto en análisis de datos con Python y pandas.

Ya tienes cargado un DataFrame llamado 'df' que contiene los datos del CSV.

Genera exclusivamente el código en Python usando el DataFrame 'df' para responder esta pregunta:

\"{pregunta}\"

No expliques nada, no leas archivos CSV ni declares variables para cargar datos. Solo responde con un bloque de código Python entre triple backticks, usando solo 'df'.
"""

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": prompt}],
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

        match = re.search(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
        code_str = match.group(1).strip() if match else answer.strip()

        print("Código generado por el modelo:\n", code_str)

        local_vars = {'df': df_cache.copy()}
        output = io.StringIO()

        try:
            with contextlib.redirect_stdout(output):
                exec(code_str, {}, local_vars)

            if 'result' in local_vars:
                result = local_vars['result']
            else:
                result = output.getvalue().strip()  # Lo que se imprimió

        except Exception as e:
            return {
                "codigo_python": code_str,
                "error": f"Error al ejecutar el código: {str(e)}"
            }

        # Convertir a JSON si es posible
        if hasattr(result, "to_dict"):
            result_json = result.to_dict(orient="records")
        else:
            result_json = str(result)

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
