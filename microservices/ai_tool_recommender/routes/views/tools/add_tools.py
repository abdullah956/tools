"""Add tools endpoint."""

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile

from microservices.ai_tool_recommender.ai_agents.excel_handler import process_excel
from microservices.ai_tool_recommender.db.database import get_add_database

router = APIRouter()


async def process_file_async(file_contents: bytes, table):
    """Asynchronous function to process the Excel file contents and add records."""
    # Now process_excel is async, so we can await it directly
    records = await process_excel(file_contents, table)
    return records


@router.post("/add_tools/")
async def upload_excel(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    table=Depends(get_add_database),
):
    """Upload and process an Excel file containing AI tools data.

    Args:
        file: Excel file containing AI tools data
        background_tasks: FastAPI background tasks
        table: Database table dependency

    Returns:
        Success message with number of records processed
    """
    try:
        # Read file contents
        file_contents = await file.read()

        # Process file asynchronously
        records = await process_file_async(file_contents, table)

        return {
            "message": f"Successfully processed {len(records)} records from {file.filename}",
            "filename": file.filename,
            "records_count": len(records),
        }

    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}
