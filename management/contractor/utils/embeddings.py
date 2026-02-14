"""Utility functions for generating embeddings."""

import logging
from functools import lru_cache
from typing import List

from envs.env_loader import env_loader
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=env_loader.openai_api_key)

# Initialize LangChain embeddings model for potential batch operations
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", api_key=env_loader.openai_api_key
)


@lru_cache(maxsize=1000)
def get_embedding(text: str) -> List[float]:
    """Get embedding for the given text using OpenAI's API.

    Args:
        text (str): The text to generate embeddings for

    Returns:
        List[float]: The embedding vector

    Raises:
        Exception: If embedding generation fails
    """
    try:
        # Clean and prepare the text
        text = text.strip().replace("\n", " ")

        # Use direct OpenAI client for single embeddings
        response = client.embeddings.create(model="text-embedding-ada-002", input=text)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts in batch using LangChain.

    Args:
        texts (List[str]): List of texts to generate embeddings for

    Returns:
        List[List[float]]: List of embedding vectors

    Raises:
        Exception: If batch embedding generation fails
    """
    try:
        # Clean and prepare the texts
        texts = [text.strip().replace("\n", " ") for text in texts]

        # Use LangChain for batch operations
        embeddings = embeddings_model.embed_documents(texts)
        return embeddings
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {str(e)}")
        raise
