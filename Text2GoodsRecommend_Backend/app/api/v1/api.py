# app/api/v1/api.py
from fastapi import APIRouter
from app.api.v1.endpoints import message_router,test_router

api_router = APIRouter()

api_router.include_router(message_router, prefix="/message", tags=["message"])
api_router.include_router(test_router, prefix="/test", tags=["test"])