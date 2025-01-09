# main.py
from fastapi import FastAPI
from api.routes import router  # Ensure this imports your router correctly
import os

# Create the main FastAPI app
app = FastAPI(title="Sunlink AI Engine")

# Include the router
app.include_router(router)

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