FROM python:3.8.3-alpine

ENV STATIC_URL /static

ENV FLASK_APP app.py

ENV FLASK_RUN_HOST 0.0.0.0

RUN apk add --no-cache gcc musl-dev linux-headers

COPY . .

RUN pip install -r requirements.txt

CMD ["flask", "run"]
