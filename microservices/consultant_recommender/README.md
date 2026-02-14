# Consultant Recommender Microservice

This microservice provides consultant recommendations based on user queries and work descriptions. It uses FastAPI, LangChain, and OpenAI to provide intelligent consultant matching.

## Features

- Search for consultants based on natural language queries
- Match consultants based on user work descriptions
- Add new consultants to the database

## Setup

This project uses Poetry for dependency management:

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Run the development server
uvicorn main:app --reload
```

## API Endpoints

- `/consultant_recommender/search_consultant/`: Search for consultants
- `/consultant_recommender/add_consultant/`: Add new consultants to the database

## Development

This project is configured with development tools:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- pytest for testing

## Docker

```bash
docker compose -f docker-compose.dev.yml up --build
```

## Install dependencies

```bash
docker-compose -f docker-compose.dev.yml run --rm consultant_recommender poetry add pyarrow
```

## Run the development server

```bash
docker-compose -f docker-compose.dev.yml up --build
```
