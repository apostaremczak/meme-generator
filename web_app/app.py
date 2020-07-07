import os
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate/<category_id>', methods=["GET"])
def generate_meme(category_id: str):
    images = {
        "0": "https://i.imgflip.com/47hbo7.jpg",
        "21": "https://i.imgflip.com/422mns.jpg",
        "47": "https://i.imgflip.com/47dpwj.jpg"
    }
    return images[category_id]


if __name__ == '__main__':
    app.run()
