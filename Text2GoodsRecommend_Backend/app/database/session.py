from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .connect import DATABASE_URL
'''连接数据库，管理会话'''

# 创建数据库引擎
Engine = create_engine(DATABASE_URL)

# 创建SessionLocal类
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=Engine)

# 依赖项，用于获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()