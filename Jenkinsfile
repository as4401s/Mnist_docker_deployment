pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // This will check out the code from your linked repository
                git 'https://github.com/as4401s/LangGraph.git'
            }
        }
        stage('Build') {
            steps {
                echo 'Building the Docker image...'
                sh 'docker-compose build'
            }
        }
        stage('Test') {
            steps {
                echo 'Running tests...'
                // Run the test.py script inside a temporary container
                sh 'docker-compose run --rm web python test.py'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying the application...'
                // Run the application in the background (detached mode)
                sh 'docker-compose up -d'
            }
        }
    }
    post {
        always {
            // This block runs regardless of the pipeline's success or failure
            echo 'Pipeline finished. Cleaning up...'
            // Stop and remove the containers to keep the environment clean
            sh 'docker-compose down'
        }
    }
}