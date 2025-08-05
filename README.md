# MNIST Digit Recognizer API with PyTorch, Docker, and Jenkins

This project demonstrates a complete end-to-end workflow for training a deep learning model, wrapping it in a web API, and deploying it using a fully automated CI/CD pipeline. The application trains a MobileNetV2 model on the classic MNIST dataset and exposes it through a Flask API, containerized with Docker and deployed with Jenkins.

![MNIST App Screenshot](https://placehold.co/800x400/007bff/ffffff?text=MNIST+App+UI)

---

## Features

-   **Model Training**: A Python script (`main.py`) to train a PyTorch MobileNetV2 model on the MNIST dataset.
-   **Web Interface**: A user-friendly front-end (`index.html`) where you can draw a digit and get a real-time prediction.
-   **REST API**: A robust Flask backend (`app.py`) that serves the model through a `/predict` endpoint.
-   **Containerization**: A `Dockerfile` and `docker-compose.yml` to create a portable and reproducible environment for the application.
-   **Automated CI/CD**: A `Jenkinsfile` that defines a complete pipeline to automatically build and deploy the application upon code changes.

---

## 🚀 Tech Stack

-   **Backend**: Python, Flask
-   **Deep Learning**: PyTorch, Torchvision
-   **Containerization**: Docker, Docker Compose
-   **CI/CD**: Jenkins
-   **Frontend**: HTML, CSS

---

## 📂 Project Structure


├── static/ <br>
│   └── style.css         # Styles for the web interface <br>
├── templates/ <br>
│   └── index.html        # Frontend HTML page <br>
├── app.py                # Flask application with the API <br>
├── main.py               # Script to train the model <br>
├── test.py               # Script to test inference locally <br>
├── Dockerfile            # Instructions to build the Docker image <br>
├── docker-compose.yml    # Defines how to run the container <br>
├── Jenkinsfile           # CI/CD pipeline script <br>
├── mnist_mobilenet.pth   # (Generated after training) <br>
└── requirements.txt      # Python dependencies <br>


---

## 🏁 Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

-   [Git](https://git-scm.com/)
-   [Docker](https://www.docker.com/products/docker-desktop/)
-   [Docker Compose](https://docs.docker.com/compose/install/)

### Local Installation & Setup

**1. Clone the Repository**

git clone https://github.com/as4401s/Mnist_docker_deployment.git <br>
cd ..

**2. Train the Model**

Before building the Docker image, you need to train the model. Run the training script, which will create the mnist_mobilenet.pth file.

###### Make sure you have Python and the dependencies from requirements.txt installed locally
pip install -r requirements.txt
python main.py

**3. Build and Run with Docker Compose**

This single command reads the docker-compose.yml file, builds the image from the Dockerfile, and starts the container.

docker-compose up --build

**4. Access the Application**

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