from random import choice
from flask import Flask, render_template, jsonify

from model_api import get_model_api

app = Flask(__name__)
model_api = get_model_api()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate/<category_id>', methods=["GET"])
def generate_meme(category_id: str):
    return jsonify(result_image=model_api(category_id))


if __name__ == '__main__':
    app.run()
