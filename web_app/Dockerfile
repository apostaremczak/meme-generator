FROM tensorflow/tensorflow:2.2.0

ENV STATIC_URL /static

ENV FLASK_APP app.py

ENV FLASK_RUN_HOST 0.0.0.0

RUN apt-get update

RUN apt-get install gcc musl-dev

COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["flask", "run"]
