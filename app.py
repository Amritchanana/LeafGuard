import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = load_model('model_savedmodel')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getResult(image_path):
    img = load_img(image_path, target_size=(225,225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"

    f = request.files['file']

    if f.filename == '':
        return "No selected file"
    
    if not allowed_file(f.filename):
        return "Invalid file type"
    
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(file_path)

    try:
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        os.remove(file_path)  # cleanup
        return str(predicted_label)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return f"Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)