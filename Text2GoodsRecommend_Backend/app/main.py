from fastapi import FastAPI,Depends,HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import UserData
from app.core import settings
from app.api.v1 import api_router


def create_application()->FastAPI:
    application = FastAPI()
    
    # 配置 CORS 允许 Vue 前端访问
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,  # Vue 开发服务器地址
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )
    application.include_router(
        api_router,
        prefix=settings.API_V1_STR
    )
    return application


app = create_application()


@app.get("/")
def root():
    return{
        "message":"success"
    }


@app.post("/data")
def receive_data(data:UserData):
    # 处理数据：比如计算是否成年
    is_adult = data.age >= 18
    
    # 返回处理后的结果
    result = {
        "name": data.name,
        "age": data.age,
        "email": data.email,
        "is_adult": is_adult,
        "status": "processed"
    }
    
    return result



'''
@app.post("/content")
async def receive_content(content_data:RequestContent,db:Session = Depends(get_db)):
    #接收Json数据并存入数据库
    try:
        #创建数据库模型实例
        db_content = Content(
            content=content_data.content,
            date=content_data.date
        )
        
        #添加到数据库
        db.add(db_content)
        db.commit()
        db.refresh(db_content)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"保存数据时出错: {str(e)}")
'''