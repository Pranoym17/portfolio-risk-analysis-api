from fastapi import FastAPI
from .database import engine
from . import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Portfolio Risk Analysis API")

@app.get("/")
def root():
    return {"status": "ok", "message": "Portfolio Risk API Running"}