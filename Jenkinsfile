// Jenkinsfile

// Defines the entire pipeline.
pipeline {
    // Specifies that the pipeline can run on any available Jenkins agent.
    agent any

    // Defines the different stages of the pipeline.
    stages {
        // Stage 1: Get the code from version control.
        stage('Checkout Source') {
            steps {
                echo 'Checking out code from repository...'
                // Replace with your actual Git repository URL.
                // This step clones your project into the Jenkins workspace.
                git 'https://github.com/your-username/mnist-deployment.git'
            }
        }

        // Stage 2: Build the Docker image.
        stage('Build Docker Image') {
            steps {
                echo 'Building the Docker image...'
                // Executes the 'docker-compose build' command.
                // --no-cache ensures we build from scratch with the latest code.
                sh 'docker-compose build --no-cache'
            }
        }

        // Stage 3: Deploy the application.
        stage('Deploy Application') {
            steps {
                echo 'Deploying the container...'
                // 'docker-compose down' stops and removes any existing container
                // to prevent conflicts.
                sh 'docker-compose down'
                // 'docker-compose up -d' starts the new container in detached mode,
                // meaning it runs in the background.
                sh 'docker-compose up -d'
            }
        }
    }

    // This block runs after all stages are complete, regardless of success or failure.
    post {
        always {
            echo 'Cleaning up old Docker images...'
            // 'docker image prune -f' removes unused (dangling) Docker images
            // to free up disk space on the Jenkins server.
            sh 'docker image prune -f'
        }
    }
}
