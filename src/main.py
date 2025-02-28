from fastapi import FastAPI 
from src.search.router import router
app = FastAPI()

app.include_router(router, prefix="/nlp")

@app.get("/")
async def welcome():
    return {"message" : "welcome"}

