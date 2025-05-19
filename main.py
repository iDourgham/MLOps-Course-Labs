## Entry Point for the FastAPI application

import logging
import os
from fastapi import FastAPI, HTTPException
from src.api import app

os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    filename="logs/churn_api.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)