<h3 align="center">Webpage-RAG-LOCAL</h3>


<!-- ABOUT THE PROJECT -->
## About The Project

This project enables users to interact with web pages through a chatbot interface, utilizing Retrieval-Augmented Generation (RAG) techniques. The application is built with Streamlit and integrates local models for processing.

**Features**
1. Webpage Interaction: Chat with any webpage by providing its URL.
2. Local Processing: Utilizes local models to ensure data privacy and reduce latency.
3. Flexible Deployment: Run the application locally, within a Docker container, or on a Kubernetes cluster.



## Prerequisites

Python 3.8 or higher
Docker (for containerized deployment)
Kubernetes (for cluster deployment)

### Installation

1. Clone the Repository
   
```sh
git clone https://github.com/SrujanTopalle/Webpage-Rag-LOCAL.git
cd Webpage-Rag-LOCAL
```

2. Install Dependencies
It's recommended to use a virtual environment:

```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
pip install -r requirements.txt
```

3. Running the Application:
   
# Option 1: Run Locally

```sh
streamlit run app_local.py
```

To access the Interface navigate to http://localhost:8501 in your web browser.

# Option 2: Run with Docker

```sh
docker build -t webpage-rag-local .
```
```sh
docker run -p 8501:8501 webpage-rag-local
```

ELSE: PULL DOCKER IMAGE FROM DOCKER HUB
```sh
docker pull srujantopalle/devopsproject
```

To access the Interface navigate to http://localhost:8501 in your web browser.

# Option 3: Deploy on Kubernetes

Verify that your Kubernetes cluster is active and kubectl is configured.
Build and Push Docker Image:
Tag the Image:

```sh
docker tag webpage-rag-local yourdockerhubusername/webpage-rag-local:latest
Push to Docker Hub:
```
```sh
docker push yourdockerhubusername/webpage-rag-local:latest
```
```sh
kubectl apply -f Kubernetes/deployment.yaml
kubectl apply -f Kubernetes/service.yaml
```
```sh
kubectl get services
```

Once the service is up, access the application using the external IP provided by the LoadBalancer.
