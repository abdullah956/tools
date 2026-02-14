"""Configuration module for loading environment variables."""

from envs.env_loader import EnvLoader

env_loader = EnvLoader()

OPENAI_API_KEY = env_loader.openai_api_key
OPENAI_MODEL = env_loader.openai_model
PINECONE_API_KEY = env_loader.pinecone_api_key

print("OPENAI_API_KEY", OPENAI_API_KEY)
print("OPENAI_MODEL", OPENAI_MODEL)
print("PINECONE_API_KEY", PINECONE_API_KEY)
