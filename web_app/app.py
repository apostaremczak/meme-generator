from random import choice
from flask import Flask, render_template, jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate/<category_id>', methods=["GET"])
def generate_meme(category_id: str):
    images = [
        "//i.imgflip.com/47hbo7.jpg",
        "//i.imgflip.com/422mns.jpg",
        "//i.imgflip.com/47dpwj.jpg"
    ]
    return jsonify(result_image=choice(images))


if __name__ == '__main__':
    app.run()
