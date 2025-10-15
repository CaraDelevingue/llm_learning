from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # 基础配置
    PROJECT_NAME: str = "Text2GoodsRecommend_Backend"
    API_V1_STR: str = "/api/v1"
    
    # CORS 配置
    # Vue 开发服务器地址
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]


settings = Settings()