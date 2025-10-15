from pydantic import BaseModel
from datetime import datetime


class UserData(BaseModel):
    name:str
    age:int
    email:str

'''查询参数模型'''
class RequestContent(BaseModel):
    content:str
    date:datetime
    
'''响应模型'''    
class ResponseContent(BaseModel):
    status:str="failure"
    message:str
    date:datetime
    


    