# run.py (项目根目录)
import uvicorn
import os,sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # 开发环境启用热重载
        log_level="info"
    )