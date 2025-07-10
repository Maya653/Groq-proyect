#se indica la imagen de referencia
FROM ubuntu:latest

#Directorio de trabajo
WORKDIR /Modelo2
RUN apt-get update
RUN apt-get install python3-full -y
RUN apt-get install -y python3-pip

COPY . /Modelo2

RUN python3 -m venv .venv

RUN .venv/bin/pip install -r Modelo2/requirements.txt
#RUN .venv/bin/pip install web.py
EXPOSE 8080

CMD [".venv/bin/python3","app/app.py","8080"]
