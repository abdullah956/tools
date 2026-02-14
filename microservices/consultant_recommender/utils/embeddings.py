"""Utility module for generating text embeddings using OpenAI."""

from langchain_openai import OpenAIEmbeddings

from envs.env_loader import EnvLoader

env_loader = EnvLoader()

embeddings_model = OpenAIEmbeddings(
    model=env_loader.openai_model, api_key=env_loader.openai_api_key
)


def get_embedding(text: str):
    """Generate an embedding for the given text."""
    return embeddings_model.embed_query(text)
