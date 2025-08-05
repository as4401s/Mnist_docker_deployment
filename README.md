MNIST Digit Recognizer API with PyTorch, Docker, and Jenkins
This project demonstrates a complete end-to-end workflow for training a deep learning model, wrapping it in a web API, and deploying it using a fully automated CI/CD pipeline. The application trains a MobileNetV2 model on the classic MNIST dataset and exposes it through a Flask API, containerized with Docker and deployed with Jenkins.

Features
Model Training: A Python script (main.py) to train a PyTorch MobileNetV2 model on the MNIST dataset.

Web Interface: A user-friendly front-end (index.html) where you can draw a digit and get a real-time prediction.

REST API: A robust Flask backend (app.py) that serves the model through a /predict endpoint.

Containerization: A Dockerfile and docker-compose.yml to create a portable and reproducible environment for the application.

Automated CI/CD: A Jenkinsfile that defines a complete pipeline to automatically build and deploy the application upon code changes.

🚀 Tech Stack
Backend: Python, Flask

Deep Learning: PyTorch, Torchvision

Containerization: Docker, Docker Compose

CI/CD: Jenkins

Frontend: HTML, CSS, JavaScript

📂 Project Structure
.
├── static/
│   └── style.css         # Styles for the web interface
├── templates/
│   └── index.html        # Frontend HTML page
├── app.py                # Flask application with the API
├── main.py               # Script to train the model
├── test.py               # Script to test inference locally
├── Dockerfile            # Instructions to build the Docker image
├── docker-compose.yml    # Defines how to run the container
├── Jenkinsfile           # CI/CD pipeline script
├── mnist_mobilenet.pth   # (Generated after training)
└── requirements.txt      # Python dependencies

🏁 Getting Started
Follow these instructions to get the project up and running on your local machine.

Prerequisites
Git

Docker

Docker Compose

Local Installation & Setup
1. Clone the Repository

git clone [https://github.com/your-username/mnist-deployment.git](https://github.com/your-username/mnist-deployment.git)
cd mnist-deployment

2. Train the Model

Before building the Docker image, you need to train the model. Run the training script, which will create the mnist_mobilenet.pth file.

# Make sure you have Python and the dependencies from requirements.txt installed locally
pip install -r requirements.txt
python main.py

3. Build and Run with Docker Compose

This single command reads the docker-compose.yml file, builds the image from the Dockerfile, and starts the container.

docker-compose up --build

4. Access the Application

Once the container is running, open your web browser and navigate to:
http://localhost:5001

You should see the web interface. Draw a digit and test the prediction!

⚙️ CI/CD Pipeline with Jenkins
This project includes a Jenkinsfile to create a fully automated deployment pipeline.

How It Works
Trigger: The pipeline is triggered automatically when you push a new commit to the main branch of your Git repository.

Checkout: Jenkins checks out the latest version of your code.

Build: It runs docker-compose build --no-cache to create a fresh Docker image with the latest changes.

Deploy: It stops the old container (docker-compose down) and starts a new one from the newly built image (docker-compose up -d).

Cleanup: After the pipeline finishes, it prunes old, unused Docker images to save disk space.

This setup ensures that any changes you push are automatically deployed, creating a seamless and efficient workflow.

🔌 API Endpoint
You can also interact with the model directly through its API endpoint.

URL: /predict

Method: POST

Body: form-data with a single key file containing the image.

Example with curl
# Replace 'path/to/your/digit.png' with an actual image file
curl -X POST -F "file=@path/to/your/digit.png" http://localhost:5001/predict

Expected Response:

{
  "prediction": 7
}