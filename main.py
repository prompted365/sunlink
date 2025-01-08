# main.py
from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Sunlink AI Engine")

# Include the router
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Hello from Sunlink Solar AI Engine!"}

if __name__ == "__main__":
    """
    Optional local-development entry point using uvicorn.
    Typically you'd rely on Gunicorn in production.
    """
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)