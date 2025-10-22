# Deployment Guide for Privacy-Preserving Synthetic Data Generator

This guide provides instructions for deploying the Privacy-Preserving Synthetic Data Generator application to various platforms.

## Option 1: Streamlit Cloud (Recommended)

The easiest way to deploy this Streamlit application is using Streamlit Cloud.

### Prerequisites

- A GitHub account
- Your code pushed to a GitHub repository (already completed)

### Deployment Steps

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `gkganesh12/Privacy-Preserving-Synthetic-Data-Generator`
5. Select the branch: `main`
6. Set the main file path: `app.py`
7. Click "Deploy"

Streamlit Cloud will automatically detect your requirements.txt file and install the necessary dependencies.

## Option 2: Heroku

### Prerequisites

- A Heroku account
- Heroku CLI installed

### Deployment Steps

1. Create a `Procfile` in your project root with the following content:
   ```
   web: streamlit run app.py --server.port=$PORT
   ```

2. Login to Heroku CLI:
   ```bash
   heroku login
   ```

3. Create a new Heroku app:
   ```bash
   heroku create privacy-preserving-synthetic-data
   ```

4. Push your code to Heroku:
   ```bash
   git push heroku main
   ```

5. Open your app:
   ```bash
   heroku open
   ```

## Option 3: Docker

For more control over your deployment environment, you can use Docker.

### Prerequisites

- Docker installed on your system

### Deployment Steps

1. Create a `Dockerfile` in your project root with the following content:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py"]
   ```

2. Build the Docker image:
   ```bash
   docker build -t synthetic-data-generator .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 8501:8501 synthetic-data-generator
   ```

4. Access the application at http://localhost:8501

## Option 4: Local Deployment

You can also run the application locally for testing or personal use.

### Prerequisites

- Python 3.7+ installed

### Deployment Steps

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Access the application at http://localhost:8501

## Important Notes

- The application requires significant computational resources, especially when generating synthetic data with differential privacy enabled.
- For production deployments, consider using a service with adequate CPU and memory resources.
- Some deployment platforms may have timeout limits that could affect long-running data generation processes.