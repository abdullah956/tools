"""Routes for adding consultants to the database."""

import asyncio
import uuid

from db.database import get_add_database
from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.embeddings import get_embedding

from microservices.shared.authentication.main import verify_token

router = APIRouter()


def generate_consultant_record(row, embedding):
    """Generate a consultant record from a row and embedding."""
    return {
        "id": str(uuid.uuid4()),
        "embedding": str(embedding),
        "unnamed_0": row.get("Unnamed: 0", "N/A"),
        "date": row.get("Date", "N/A"),
        "time": row.get("Time", "N/A"),
        "company_name": row.get("Company Name", "N/A"),
        "country": row.get("Country", "N/A"),
        "apps_included": row.get("Apps Included", "N/A"),
        "language": row.get("Language", "N/A"),
        "phone": row.get("Phone", "N/A"),
        "website": row.get("Website", "N/A"),
        "gmail": row.get("Gmail", "N/A"),
        "about": row.get("About", "N/A"),
        "type_of_services": row.get("Type of services", "N/A"),
        "countries_with_office_locations": row.get(
            "Countries with Office Locations", "N/A"
        ),
    }


async def process_consultant_file_async(file_contents: bytes, db):
    """Process the Excel file contents and add consultant records."""
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, pd.read_excel, file_contents)
    df.columns = df.columns.str.strip()
    df = df.astype(str).fillna("N/A")

    df = df.head(100)

    vectors = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Consultants"):
        embedding = get_embedding(row["Type of services"])
        embedding = np.array(embedding, dtype=np.float32).tolist()
        consultant_record = generate_consultant_record(row, embedding)
        vectors.append(
            {
                "id": consultant_record["id"],
                "values": embedding,
                "metadata": {
                    key: value
                    for key, value in consultant_record.items()
                    if key != "embedding"
                },
            }
        )

    # Upsert vectors to Pinecone
    db.upsert(vectors=vectors, namespace="consultants")


@router.post("/add_consultant/")
async def add_consultant(
    file: UploadFile = File(..., max_size=1024 * 1024 * 100),  # 100MB limit
    db=Depends(get_add_database),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: dict = Depends(verify_token),
):
    """Endpoint to add consultants from an uploaded file."""
    # Read file contents directly into memory
    contents = await file.read()

    # Process the file and add records to Pinecone
    background_tasks.add_task(process_consultant_file_async, contents, db)

    return {
        "user": current_user,
        "message": "✅ Consultant file processing started in background!",
    }
