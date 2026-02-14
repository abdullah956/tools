"""Database for the AI tool recommender."""

from pinecone import Pinecone, ServerlessSpec

from envs.env_loader import EnvLoader

env_loader = EnvLoader()

# Create an instance of the Pinecone class
pinecone_client = Pinecone(api_key=env_loader.pinecone_api_key)
print("pinecone_client", pinecone_client)
print("env_loader.pinecone_api_key", env_loader.pinecone_api_key)
# Check if the index exists, if not, create it
index_name = env_loader.pinecone_tool_index
print(index_name)
try:
    indexes = pinecone_client.list_indexes()
    print("indexes", indexes)
    if index_name not in [idx.name for idx in indexes]:
        pinecone_client.create_index(
            name=index_name,
            dimension=1536,  # Assuming the embedding size is 1536
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
except Exception as e:
    print(f"Error with Pinecone: {e}")

# Connect to Pinecone index
try:
    pinecone_index = pinecone_client.Index(index_name)
except Exception as e:
    print(f"Error connecting to Pinecone index: {e}")
    pinecone_index = None


def get_search_database():
    """Returns the Pinecone index for search operations."""
    return pinecone_index


def get_add_database():
    """Returns the Pinecone index for adding/updating operations."""
    return pinecone_index
