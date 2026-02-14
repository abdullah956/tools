# AI Tool Recommender Microservice

## Overview

The AI Tool Recommender is a specialized microservice that helps users discover relevant AI tools based on their specific needs and requirements. It leverages vector embeddings and semantic search to match natural language queries with the most appropriate AI tools from our database.

This microservice is built with:
- **FastAPI**: For creating a high-performance REST API
- **LangChain**: For AI/LLM orchestration and embeddings
- **OpenAI**: For generating embeddings and enhancing search results
- **Vector Database**: For efficient similarity search of AI tools

## Features

- Search for AI tools based on natural language queries
- Get detailed explanations of recommended tools
- Add new tools to the database
- Authentication via JWT tokens
- RESTful API design

## API Endpoints

- `/ai_tool_recommender/search_tools/`: Search for AI tools based on user queries
- `/ai_tool_recommender/explain/`: Get detailed explanations for specific tools
- `/ai_tool_recommender/add_tools/`: Add new tools to the database

## Local Development Setup

This project uses Poetry for dependency management:

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Run the development server
uvicorn main:app --reload --host 0.0.0.0 --port 12000
```

## Docker Setup

### Development Environment

To start the service in development mode:

```bash
# Build the development Docker image
docker build -f Dockerfile.dev -t ai-tool-recommender-dev .

# Run the container
docker run -p 12000:12000 \
  -v $(pwd):/app/microservices/ai_tool_recommender \
  --name ai-tool-recommender-dev \
  ai-tool-recommender-dev
```

The development container includes hot-reloading, so any changes to the code will automatically restart the server.

### Staging Environment

To start the service in staging mode:

```bash
# Build the staging Docker image
docker build -f Dockerfile.staging -t ai-tool-recommender-staging .

# Run the container
docker run -p 12000:12000 \
  --name ai-tool-recommender-staging \
  ai-tool-recommender-staging
```

The staging container includes:
- Multi-stage build for smaller image size
- Waits for the Management service to be available before starting
- Uses staging environment variables from .env.staging

## Development Tools

This project is configured with development tools:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- pytest for testing

## API Documentation

When the service is running, you can access the API documentation at:
- Swagger UI: http://localhost:12000/ai_tool_recommender/docs
- ReDoc: http://localhost:12000/ai_tool_recommender/redoc
