# main.py
from fastapi import FastAPI, APIRouter
import os
from fastapi import APIRouter
from pydantic import BaseModel
from tasks.worker_tasks import process_solar_task

# Create the main FastAPI app
app = FastAPI(title="Sunlink AI Engine")

# Define the Pydantic schema for the POST /process payload
class TaskPayload(BaseModel):
    property_id: str

@app.post("/process")
def enqueue_solar_job(payload: TaskPayload):
    """
    Endpoint to queue a background solar processing job.
    Expects JSON: {"property_id": "some-uuid-or-id"}
    """
    process_solar_task.delay(payload.property_id)
    return {"status": "enqueued", "property_id": payload.property_id}

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Root endpoint
@app.get("/")
def root():
    return {"message": "Hello from the Sunlink.ai Engine! Put power back in your hands!"}

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)