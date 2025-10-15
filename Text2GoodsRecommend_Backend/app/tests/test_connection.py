from sqlalchemy import text
from app.database import SessionLocal


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