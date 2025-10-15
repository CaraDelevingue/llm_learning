import os
from dotenv import load_dotenv
'''数据库连接配置'''

#获取项目根目录
ROOT_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#获取.env配置文件目录
DOTENV_PATH=os.path.join(ROOT_DIR,'.env')
#加载.env配置文件
load_dotenv(DOTENV_PATH)

# 配置数据库连接参数
username=os.getenv("DB_USER")
password=os.getenv("DB_PASSWORD")
host=os.getenv("DB_HOST","localhost")
port=os.getenv("DB_POST","5432")
database=os.getenv("DB_NAME")

#检查无默认值参数非空
if not all([username, password, database]):
    raise EnvironmentError("Missing database environment variables")

#构建数据库连接URL
DATABASE_URL = f"postgresql://{username}:{password}@{host}:{port}/{database}"