from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import base64

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'Data'
MODEL_PATH = 'model.pkl'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Helper functions
def load_images_from_folder(folder):
    """Loads and preprocesses images from a folder."""
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                try:
                    img = Image.open(img_path).convert('L')  # Grayscale
                    img = img.resize((64, 64))  # Resize
                    images.append(np.array(img).flatten())
                    labels.append(label)
                except Exception as e:
                    print(f"Could not load image {img_path}: {e}")
    return np.array(images), np.array(labels)

def train_model():
    """Trains a k-NN model and saves it."""
    print("Training model...")
    images, labels = load_images_from_folder(DATA_FOLDER)
    if images.size == 0:
        return {"error": "No data found to train the model."}

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    return {
        "message": "Model trained and saved successfully!",
        "train_accuracy": f"{model.score(X_train, y_train):.2f}",
        "test_accuracy": f"{model.score(X_test, y_test):.2f}"
    }

def classify_image(image_path):
    """Loads a saved model and classifies a new image."""
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not found. Please train the model first."}
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    img = Image.open(image_path).convert('L')
    img = img.resize((64, 64))
    img_array = np.array(img).flatten().reshape(1, -1)

    prediction = model.predict(img_array)
    return {"prediction": prediction[0]}

# Routes
@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    """Endpoint to train the model."""
    if not os.path.exists(DATA_FOLDER) or not os.listdir(DATA_FOLDER):
        return jsonify({"error": f"Data folder '{DATA_FOLDER}' does not exist or is empty."}), 400
    
    result = train_model()
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result), 200

@app.route('/classify', methods=['POST'])
def classify():
    """Endpoint to classify an uploaded image."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        prediction_result = classify_image(filename)
        os.remove(filename)  # Clean up the uploaded file
        
        if "error" in prediction_result:
            return jsonify(prediction_result), 400
        
        # Read the file again to get bytes for base64 encoding
        # This is necessary because file.read() was used when saving
        file.seek(0)
        img_bytes = file.read()
        base64_encoded_img = base64.b64encode(img_bytes).decode('utf-8')
        
        return jsonify({
            "message": "Image classified successfully!",
            "prediction": prediction_result['prediction'],
            "image": base64_encoded_img,
            "image_type": file.mimetype
        }), 200
        
if __name__ == '__main__':
    app.run(debug=True)