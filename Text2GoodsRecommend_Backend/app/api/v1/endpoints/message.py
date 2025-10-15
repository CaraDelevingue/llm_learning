from fastapi import APIRouter
from app.schemas import RequestContent,ResponseContent
from app.services import model_service

router = APIRouter(prefix="/message", tags=["message"])
   

# 接收 JSON 消息的端点
@router.post("/",response_model=ResponseContent)
async def receive_message(content_data:RequestContent):
    print(f"收到消息: {content_data.content} ")
    
    result = model_service(content_data.content)
    # 处理逻辑...
    response_data = {
        "status": "success",
        "message": result,
        "date": content_data.date
    }
    
    return response_data
