"""Excel handler for the AI tool recommender."""

import asyncio
import uuid

import pandas as pd
from pinecone import Index
from tqdm import tqdm

from ai_tool_recommender.utils.embeddings import get_embedding


async def process_excel(file_contents: bytes, db: Index):
    """Process an Excel file and upload data to Pinecone."""
    df = pd.read_excel(file_contents)
    df = df.astype(str).fillna("N/A")

    df = df.head(100)

    vectors = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        description = row.get("Description", "").strip()
        if not description:
            raise ValueError("Description is required for each row.")

        embedding = await get_embedding(description)
        record_id = str(uuid.uuid4())

        # Prepare vector for Pinecone
        vectors.append(
            {"id": record_id, "values": embedding, "metadata": row.to_dict()}
        )

    # Upsert vectors to Pinecone (run in thread pool for async)
    await asyncio.to_thread(db.upsert, vectors=vectors, namespace="ai_tools")

    return len(df)
