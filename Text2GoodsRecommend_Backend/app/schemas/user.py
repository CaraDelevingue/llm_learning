from pydantic import BaseModel
from datetime import datetime


class UserData(BaseModel):
    name:str
    age:int
    email:str


class RequestContent(BaseModel):
    content:str
    date:datetime


    