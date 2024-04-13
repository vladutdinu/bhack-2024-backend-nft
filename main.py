

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routers.nft_check_route import nft_check_route

# Load environment variables from .env
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
    "https://chainsentinel.mihneahututui.eu",
    "https://chainsentinel.app.genez.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes

app.include_router(nft_check_route)

if __name__ == "__main__":

    uvicorn.run("main:app", host=os.environ["HOST"], port=os.environ["PORT"], log_level='info')
