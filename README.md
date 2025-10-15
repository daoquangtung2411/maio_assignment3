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
# Version 0.1
docker pull ghcr.io/daoquangtung2411/maio_assignment3/diabetes_progression:v0.1.21
# Version 0.2
docker pull ghcr.io/daoquangtung2411/maio_assignment3/diabetes_progression:v0.2.3
# Run container
# Version 0.1
docker run -d -p 8386:8386 ghcr.io/daoquangtung2411/maio_assignment3/diabetes_progression:v0.1.21
# Version 0.2
docker run -d -p 8686:8686 ghcr.io/daoquangtung2411/maio_assignment3/diabetes_progression:v0.2.3

# Option 2:
# Download docker compose file
curl -0 https://raw.githubusercontent.com/daoquangtung2411/maio_assignment3/main/docker-compose.yml -o docker-compose.yml

# (If this return error in Windows, please try:
Invoke_WebRequest -Uri "https://raw.githubusercontent.com/daoquangtung2411/maio_assignment3/main/docker-compose.yml" -OutFile "docker-compose.yml"
)
# Run container
# Version 0.1
docker compose up -d dp_production_v0.1
# Version 0.2
docker compose up -d dp_production_v0.2
```

## Retrain or reproduce locally

```bash
# To reproduce repository from scratch, follow below instructions:
# Clone the repository
git clone https://github.com/daoquangtung2411/maio_assignment3.git

# Run docker
# Version 0.1
docker compose up -d dp_retrain_v0.1
# Version 0.2
docker compose up -d dp_retrain_v0.2

```

## Run the prediction API

```bash
# Health check
# Version 0.1
curl -X GET http://localhost:8386/health
# Version 0.2
curl -X GET http://localhost:8686/health

# Prediction
# Version 0.1
curl -X POST http://localhost:8386/predict -H "Content-Type:application/json" -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'

# Version 0.2
curl -X POST http://localhost:8686/predict -H "Content-Type:application/json" -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
```

## Port

- **Model v0.1 API**: `http://localhost:8386`
- **Model v0.2 API**: `http://localhost:8686`