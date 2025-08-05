import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

def train_mnist_model():
    """
    This function trains a MobileNetV2 model on the MNIST dataset
    and saves the trained model weights.
    """
    print("--- Starting Model Training ---")

    # --- 1. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Loading and Transformation ---
    # MNIST images are 28x28 grayscale. MobileNetV2 expects 3-channel images,
    # so we will duplicate the single channel to three.
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # MobileNetV2 works well with 32x32 or larger
        transforms.Grayscale(num_output_channels=3), # Convert grayscale to 3-channel
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize for 3 channels
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)
    print("MNIST dataset loaded successfully.")

    # --- 3. Model Definition ---
    # Load a pre-trained MobileNetV2 and modify it for MNIST (10 classes)
    model = mobilenet_v2(weights='IMAGENET1K_V1')

    # Modify the classifier for 10 output classes (digits 0-9)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    
    model.to(device)
    print("MobileNetV2 model adapted for MNIST.")

    # --- 4. Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 5. Training Loop ---
    # Set epochs to 100 for full training as requested, or a smaller number for a quick test.
    num_epochs = 10 
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199: # Print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print("--- Finished Training ---")

    # --- 6. Save the Model ---
    model_path = 'mnist_mobilenet.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_mnist_model()