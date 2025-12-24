# pip install flask tensorflow numpy pillow

from flask import Flask, render_template, request, jsonify, url_for
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = tf.keras.models.load_model("best_model.h5")
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0
    return img_array

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_array = preprocess_image(filepath)
        preds = model.predict(img_array)
        class_index = np.argmax(preds, axis=1)[0]
        confidence = float(preds[0][class_index] * 100)
        prediction = class_names[class_index]

        file_size = os.path.getsize(filepath)
        image_url = url_for('static', filename=f'uploads/{filename}', _external=True)

        return jsonify({
            'prediction': prediction,
            'tumor_type': prediction,
            'confidence': round(confidence, 2),
            'image_url': image_url,
            'file_name': filename,
            'file_size': file_size
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
