FROM python:3.11

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./app /app

EXPOSE 8000

CMD uvicorn app.app:app --host 0.0.0.0 --port 8000