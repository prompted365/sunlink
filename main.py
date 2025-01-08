# main.py
from fastapi import FastAPI
from api.routes import router
import os

app = FastAPI(title="Sunlink AI Engine")

# Include the router
app.include_router(router)

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Hello from Sunlink Solar AI Engine!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)