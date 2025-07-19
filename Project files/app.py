from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- LOAD MODEL ONCE ---
model_path = os.path.join('models', 'rice.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found at: " + model_path)
model = load_model(model_path)

# --- CLASS LABELS ---
class_labels = ['Basmati', 'Jasmine', 'Arborio', 'Sona Masoori', 'Brown']

# --- ROUTES ---

@app.route('/ping')
def ping():
    return "✅ Server is working!"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part in request.'

    file = request.files['file']
    if file.filename == '':
        return 'No file selected.'

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"✅ File saved to {filepath}")

        try:
            img = image.load_img(filepath, target_size=(224, 224))
        except Exception as e:
            return f"Image loading failed: {str(e)}"

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict with timer
        start = time.time()
        predictions = model.predict(img_array)
        end = time.time()
        print(f"🔍 Prediction took {end - start:.2f} seconds")

        predicted_class = class_labels[np.argmax(predictions[0])]
        confidence = round(np.max(predictions[0]) * 100, 2)

        # Fix image path for HTML display
        image_filename = os.path.basename(filepath)
        image_path = url_for('static', filename='uploads/' + image_filename)

        return render_template('result.html',
                               prediction=predicted_class,
                               confidence=confidence,
                               image_path=image_path)

    return 'Something went wrong. Please try again.'

if __name__ == '__main__':
    app.run(debug=True)
