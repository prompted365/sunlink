# routes.py

from fastapi import APIRouter
from pydantic import BaseModel
from tasks.worker_tasks import process_solar_task

# Create a FastAPI router
router = APIRouter()

# Define the Pydantic schema for the POST /process payload
class TaskPayload(BaseModel):
    property_id: str

@router.post("/process")
def enqueue_solar_job(payload: TaskPayload):
    """
    Endpoint to queue a background solar processing job.
    Expects JSON: {"property_id": "some-uuid-or-id"}
    """
    process_solar_task.delay(payload.property_id)
    return {"status": "enqueued", "property_id": payload.property_id}
