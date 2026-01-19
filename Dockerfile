FROM python:3.11

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /code/app

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app.app:app"]