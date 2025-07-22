#se indica la imagen de referencia
FROM ubuntu:latest

#Directorio de trabajo
WORKDIR /app

RUN apt-get update
RUN apt-get install python3-full -y
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-venv  # ✅ Añadido: necesario para crear el entorno virtual

COPY . .

RUN python3 -m venv .venv

RUN .venv/bin/pip install -r requirements.txt

EXPOSE 8080

CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
