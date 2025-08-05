import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

# --- 1. Initialization ---
app = Flask(__name__)

# --- 2. Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'mnist_mobilenet.pth'
model = mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, 10) # Adapt for 10 classes

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_path}'.")
    print("The API will not work. Please run main.py to train the model.")
    model = None

if model:
    model.to(device)
    model.eval() # IMPORTANT: Set model to evaluation mode

# --- 3. Image Transformation ---
# This must match the transformation used during training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Ensure 3 channels
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- 4. API Endpoints ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives an image and returns a prediction."""
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read image bytes
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        
        # Transform the image and add a batch dimension
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make a prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            prediction = predicted.item()
            
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 5. Run the App ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible from outside the container
    app.run(host='0.0.0.0', port=5000, debug=True)