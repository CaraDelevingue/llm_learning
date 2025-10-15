from fastapi import APIRouter
from app.schemas import RequestContent,ResponseContent


router = APIRouter(prefix="/test", tags=["test"])

@router.get("/")
def test_app():
    return {
        "message":"success"
    }
 