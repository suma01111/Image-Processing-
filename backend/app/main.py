from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.labs import router as labs_router


app = FastAPI(title="Interactive Digital Image Processing Lab", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(labs_router, prefix="/api")

