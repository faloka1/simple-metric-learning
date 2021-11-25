import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import json

app = Flask(__name__)

fe = FeatureExtractor()


@app.route('/delete', methods=['POST'])
def delete():
    name = request.form['name']


@app.route('/list', methods=['GET'])
def list_image():
    files = []
    for feature_path in Path("./static/feature").glob("*.npy"):
        files.append(feature_path.stem)

    return json.dumps(files)


@app.route('/insert', methods=['POST'])
def register():
    file = request.files['img']
    name = request.form['name']

    img = Image.open(file.stream)  # PIL image
    img_path = "static/img/" + name + ".jpg"
    img.save(img_path)

    feature = fe.extract(img)
    feature_path = 'static/feature/' + name + ".npy"
    np.save(feature_path, feature)

    res = {'success': True}
    return json.dumps(res)


@app.route('/search', methods=['POST'])
def index():
    features = []
    img_paths = []
    for feature_path in Path("./static/feature").glob("*.npy"):
        features.append(np.load(feature_path))
        img_paths.append(feature_path.stem)

    features = np.array(features)

    file = request.files['img']

    # Save query image
    img = Image.open(file.stream)  # PIL image
    uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
    img.save(uploaded_img_path)

    # Run search
    query = fe.extract(img)
    dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
    ids = np.argsort(dists)[:30]  # Top 30 results
    scores = [(img_paths[id], str(dists[id])) for id in ids]

    return json.dumps(scores)


if __name__=="__main__":
    app.run("0.0.0.0", port=8080)
