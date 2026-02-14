"""Module for getting embeddings."""

from langchain_openai import OpenAIEmbeddings
import numpy as np

from envs.env_loader import EnvLoader

env_loader = EnvLoader()

embeddings_model = OpenAIEmbeddings(
    model=env_loader.openai_model, api_key=env_loader.openai_api_key
)


async def get_embedding(text: str):
    """Get the embedding for a given text."""
    import asyncio

    query_embedding = np.array(
        await asyncio.wait_for(embeddings_model.aembed_query(text), timeout=5.0),
        dtype=np.float32,
    ).tolist()
    return query_embedding
