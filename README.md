## GROUP BC - MAIO ASSIGNMENT 3 - DIABETES PROGRESSION PREDICTOR

This repository is source code for diabetes progression models, submitted as solution to assignment 3 - MAIO course 2025. The prediction model can be use with Docker, and you can run following tasks:

- Run **containerized API** with docker compose and get predictions
- Retrain model for reproducibility with docker compose

*This Docker image has been tested and worked on Windows 10, MacOS and Ubuntu*

## Prerequisites

- **Docker Desktop**
- Git (optional if you download raw file from the repo)

## Repository layout

```
maio_assignment3/
|- .github/workflows       # GitHub Actions
|- scripts                 # Train and API code
|- tests                   # Unit test
|- Dockerfile
|- README.md
|- CHANGELOG.md
|- docker-compose.yml
|- requirements.txt

```
## Quick start

```bash
# There are two ways to run the ready-to-use Docker image

# Option 1:
# Pull image
docker pull ghcr.io/daoquangtung2411/maio_assignment3/diabetes_progression:v0.1.7
# Run container
docker run -d -p 8386:8386 ghcr.io/daoquangtung2411/maio_assignment3/diabetes_progression:v0.1.7

# Option 2:
# Download docker compose file
curl -0 https://raw.githubusercontent.com/daoquangtung2411/maio_assignment3/main/docker-compose.yml -o docker-compose.yml

# (If this return error in Windows, please try:
Invoke_WebRequest -Uri "https://raw.githubusercontent.com/daoquangtung2411/maio_assignment3/main/docker-compose.yml" -OutFile "docker-compose.yml"
)
# Run container
docker compose up -d dp_production_v0.1
```

## Retrain or reproduce locally

```bash
# To reproduce repository from scratch, follow below instructions:
# Clone the repository
git clone https://github.com/daoquangtung2411/maio_assignment3.git

# Run docker
docker compose up -d dp_retrain_v0.1

```

## Run the prediction API

```bash
# Health check
curl -X GET http://localhost:8386/health

# Prediction

curl -X POST http://localhost:8386/predict -H "Content-Type:application/json" -d '{"age":10,"sex":1,"bmi":19,"bp":123,"s1":20,"s2":30,"s3":40,"s4":50,"s5":60,"s6":70}'
```

## Port

- **Model v0.1 API**: `http://localhost:8386`
- **Model v0.2 API**: `http://localhost:8686`