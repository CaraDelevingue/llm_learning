from fastapi import APIRouter,HTTPException
from app.schemas import UserCreate,UserResponse

router = APIRouter(prefix="/api", tags=["users"])

@router.post("/users", response_model=UserResponse)
async def create_user_endpoint(user: UserCreate):
    try:
        
        return True
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))