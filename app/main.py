from fastapi import FastAPI
import uvicorn

from app.api.v1.chat import router as chat_router


app = FastAPI()

app.include_router(chat_router, prefix="/api/v1", tags=["Agents"])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)