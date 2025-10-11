from sqlalchemy import create_engine,text
from sqlalchemy.orm import sessionmaker

import os
from dotenv import load_dotenv

#获取app根目录
ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#获取.env配置文件目录
DOTENV_PATH=os.path.join(ROOT_DIR,'.env')
#加载.env配置文件
load_dotenv(DOTENV_PATH)

# 数据库连接信息
username=os.getenv("DB_USER")
password=os.getenv("DB_PASSWORD")
host=os.getenv("DB_HOST","localhost")
port=os.getenv("DB_POST","5432")
database=os.getenv("DB_NAME")

if not all([username, password, database]):
    raise EnvironmentError("Missing database environment variables")

DATABASE_URL = f"postgresql://{username}:{password}@{host}:{port}/{database}"

# 创建数据库引擎
engine = create_engine(DATABASE_URL)

# 创建SessionLocal类
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



# 依赖项，用于获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
# 检测数据库连接
def test_connection():
    try:
        # 创建会话
        db = SessionLocal()
        # 执行简单的SQL查询检测连接
        result = db.execute(text("SELECT 1"))
        db.close()
        print(" 数据库连接成功!")
        return True
    except Exception as e:
        print(f" 数据库连接失败: {e}")
        return False