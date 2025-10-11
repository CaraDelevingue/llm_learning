from fastapi import FastAPI,Depends,HTTPException
from sqlalchemy.orm import Session


from schemas import RequestContent,UserData
from database.connect import engine,get_db
from database.models import metadata,Content

#启动时创建所有的表:Content
metadata.create_all(engine)

app = FastAPI()

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



@app.post("/content")
async def create_content(content_data:RequestContent,db:Session = Depends(get_db)):
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