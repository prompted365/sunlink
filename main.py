# main.py

import uvicorn
from fastapi import FastAPI
from api.routes import router  # Import the router from routes.py

app = FastAPI(title="Sunlink AI Engine")

# Include the router
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Hello from Sunlink Solar AI Engine!"}

if __name__ == "__main__":
    # Run the FastAPI server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
