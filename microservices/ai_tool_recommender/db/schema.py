"""Schema for the AI tool recommender."""

import pyarrow as pa

embedding_size = 1536  # OpenAI embedding size

schema = pa.schema(
    [
        ("id", pa.string()),
        ("title", pa.string()),
        ("description", pa.string()),
        ("category", pa.string()),
        ("features", pa.string()),
        ("tags", pa.string()),
        ("website", pa.string()),
        ("twitter", pa.string()),
        ("facebook", pa.string()),
        ("linkedin", pa.string()),
        ("embedding", pa.list_(pa.float32(), embedding_size)),  # Fixed-size embeddings
    ]
)
