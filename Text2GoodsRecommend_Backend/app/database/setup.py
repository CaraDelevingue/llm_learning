import sys
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径，确保可以导入您的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .connect import engine,test_connection
from .models import Raw,Processed


def create_raw_table():
    "创建原生数据集表"
    try:
        # 测试数据库连接
        if not test_connection():
            logger.error("无法连接到数据库，请检查连接配置")
            return False
        # 创建 Raw 表
        logger.info("正在创建数据库表...")
        Raw.__table__.create(bind=engine)
        
        logger.info(" Raw 表创建成功!")
        return True
    except Exception as e:
        logger.error(f"创建表时出错: {e}")
        return False
    
def drop_raw_table():
    """销毁原生数据集表"""
    try:
        # 测试数据库连接
        if not test_connection():
            logger.error("无法连接到数据库，请检查连接配置")
            return False
        
        print("开始销毁 Raw 表...")
        
        # 删除 Raw 表
        Raw.__table__.drop(bind=engine, checkfirst=True)
        
        print(" Raw 表已成功销毁!")
        return True  
    except Exception as e:
        print(f" 销毁表时出错: {e}")
        return False
    
    
def create_processed_table():
    "创建已处理数据集表"
    try:
        # 测试数据库连接
        if not test_connection():
            logger.error("无法连接到数据库，请检查连接配置")
            return False
        # 创建 Processed 表
        logger.info("正在创建数据库表...")
        Processed.__table__.create(bind=engine)
        
        logger.info(" Processed 表创建成功!")
        return True
    except Exception as e:
        logger.error(f"创建表时出错: {e}")
        return False
    
def drop_raw_table():
    """销毁已处理数据集表"""
    try:
        # 测试数据库连接
        if not test_connection():
            logger.error("无法连接到数据库，请检查连接配置")
            return False
        
        print("开始销毁 Processed 表...")
        
        # 删除 Processed 表
        Processed.__table__.drop(bind=engine, checkfirst=True)
        
        print(" Processed 表已成功销毁!")
        return True  
    except Exception as e:
        print(f" 销毁表时出错: {e}")
        return False
    
    
def test():
    try:
         # 测试数据库连接
        if not test_connection():
            logger.error("无法连接到数据库，请检查连接配置")
            return False
        print("连接成功")
        
    except Exception as e:
        print(f" 连接时出错: {e}")
        return False