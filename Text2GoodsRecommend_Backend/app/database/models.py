from sqlalchemy import Column, Integer, String, DateTime, Text,func,MetaData
from sqlalchemy.ext.declarative import declarative_base


# 创建metadata
metadata = MetaData()

# 创建Base类
Base = declarative_base()


class Raw(Base):
    __tablename__ = "raw"
    
    # 主键，数据库自动生成
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # 评分，1-5星
    rating = Column(Integer, nullable=False)
    
    # 评论文本，中文字符串
    review_content = Column(Text, nullable=False)
    
    # 点赞数，大于等于0
    likes = Column(Integer, default=0)
    
    # 发布时间，年月日时分秒
    publish_time = Column(DateTime(timezone=True), nullable=False)
    
    # 设备型号，中文字符串
    device_model = Column(String(100))
    
    # 游戏名称，中文字符串
    game_name = Column(String(100), nullable=False, index=True)
    
    
class Processed(Base):
    __tablename__ = "process"
    
    # 主键，数据库自动生成
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # 评分，1-5星
    rating = Column(Integer, nullable=False)
    
    # 评论文本，中文字符串
    review_content = Column(Text, nullable=False)
    
    # 点赞数，大于等于0
    likes = Column(Integer, default=0)
    
    # 发布时间，年月日时分秒
    publish_time = Column(DateTime(timezone=True), nullable=False)
    
    # 游戏名称，中文字符串
    game_name = Column(String(100), nullable=False, index=True)
    
    
class Content(Base):
    __tablename__ = "content"
    # 主键
    id = Column(Integer,primary_key=True,index=True,autoincrement=True)
    
    #Json中的content,用户询问的内容
    content = Column(Text,nullable=False)
    
    #Json中的内容，请求发送的时间
    date = Column(DateTime(timezone=True),nullable=False)
    
    #记录创建时间
    created_at = Column(DateTime(timezone=True), server_default=func.now())
