"""Routes for adding consultants."""

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile

from microservices.ai_tool_recommender.ai_agents.excel_handler import process_excel
from microservices.ai_tool_recommender.db.database import get_add_database
from microservices.shared.authentication.main import verify_token

router = APIRouter()


async def process_file_async(file_contents: bytes, table):
    """Asynchronous function to process the Excel file contents and add records."""
    loop = asyncio.get_event_loop()
    # Run process_excel in a thread pool since it's CPU-bound
    records = await loop.run_in_executor(None, process_excel, file_contents, table)
    return records


@router.post("/add_tools/")
async def upload_excel(
    file: UploadFile = File(..., max_size=1024 * 1024 * 100),  # 100MB limit
    db=Depends(get_add_database),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: dict = Depends(verify_token),
):
    """Upload an Excel file to add consultants to the database."""
    # Read file contents directly into memory
    contents = await file.read()

    # Process the file and add records to Pinecone
    background_tasks.add_task(process_file_async, contents, db)

    return {
        "user": current_user,
        "message": "✅ File processing started in background!",
    }
