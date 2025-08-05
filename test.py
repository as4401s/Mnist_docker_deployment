import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import torch.nn as nn
from PIL import Image
import random

def test_single_image():
    """
    Loads the trained model and performs inference on a single
    random image from the MNIST test set.
    """
    print("--- Running Inference Test ---")
    
    # --- 1. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Model ---
    model_path = 'mnist_mobilenet.pth'
    model = mobilenet_v2() # Create a new instance of the model
    model.classifier[1] = nn.Linear(model.last_channel, 10) # Adapt the classifier
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please run main.py first to train and save the model.")
        return

    model.to(device)
    model.eval() # Set the model to evaluation mode
    print("Model loaded successfully.")

    # --- 3. Load Test Data ---
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    
    # --- 4. Select a Random Image and Predict ---
    idx = random.randint(0, len(testset) - 1)
    image, label = testset[idx]
    
    print(f"Selected image index: {idx}")
    print(f"Actual Label: {label}")

    # Add a batch dimension (C, H, W) -> (B, C, H, W)
    image_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_batch)
        _, predicted = torch.max(output.data, 1)
        prediction = predicted.item()

    print(f"Model Prediction: {prediction}")

    if prediction == label:
        print("Result: Correct! ✅")
    else:
        print("Result: Incorrect. ❌")

if __name__ == '__main__':
    test_single_image()