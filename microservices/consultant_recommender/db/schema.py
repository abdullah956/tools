"""Schema for the consultant recommender."""

import pyarrow as pa

embedding_size = 1536  # OpenAI embedding dimension

schema = pa.schema(
    [
        ("id", pa.string()),
        (
            "embedding",
            pa.list_(pa.float32(), embedding_size),
        ),  # Assuming embeddings are variable-length lists
        ("unnamed_0", pa.string()),  # Assuming this is a string, adjust if needed
        ("date", pa.string()),  # Assuming date is stored as a string, adjust if needed
        ("time", pa.string()),  # Assuming time is stored as a string, adjust if needed
        ("company_name", pa.string()),
        ("country", pa.string()),
        ("apps_included", pa.string()),
        ("language", pa.string()),
        ("phone", pa.string()),
        ("website", pa.string()),
        ("gmail", pa.string()),
        ("about", pa.string()),
        ("type_of_services", pa.string()),
        ("countries_with_office_locations", pa.string()),
    ]
)
