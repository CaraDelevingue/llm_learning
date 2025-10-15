import os
import sys
import pandas as pd
import re
from transformers import AutoTokenizer
from sqlalchemy import  Integer, Text,String, DateTime
from sqlalchemy.orm import sessionmaker,Session
from sqlalchemy.exc import SQLAlchemyError
from zhconv import convert


def clean_reviews(text):
    """清理评论数据"""
    
    # 处理空值和类型
    if not text or not isinstance(text, str):
        return ""
 
    # 基础清理
    text = basic_cleaning(text)
    
    # 表情符号处理
    text = handle_emojis(text)
    
    # 中文特定处理
    text = chinese_specific_processing(text)
    
    # 特殊字符处理
    text = handle_special_characters(text)
    
    # 颜文字处理
    text = remove_kaomoji(text)
    
    #过滤非常规字符
    text = filter_uncommon_chars(text)
        
    # 重复字符规整
    text = normalize_repetition(text)

    # 最终清理
    text = final_cleaning(text)
    
    return text


def basic_cleaning(text):
    """基础文本清理"""
    if not text:
        return ""
    
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 统一省略号：多个省略号（包括...和…）→ ...
    text = re.sub(r'[…⋯]{2,}', '...', text)      # 多个省略号字符 → ...
    text = re.sub(r'\.{4,}', '...', text)        # 多个点 → ...
    # 统一破折号：包括 ---、___、——、——— 等 → ——（两个长破折号）
    text = re.sub(r'-{3,}', '——', text)          # 三个或更多短横线 → ——
    text = re.sub(r'_+', '——', text)             # 下划线（三个或更多）→ ——
    text = re.sub(r'—{2,}', '——', text)          # 两个或更多长破折号 → ——
    # 移除特殊控制字符但保留常见标点
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    # 统一换行符和空格
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()


def handle_emojis(text):
    """处理英文描述的emoji"""
    # 将英文问号替换为[EMOJI]
    text = re.sub(r'\?', '[EMOJI]', text)
    
    # 规整连续emoji（超过2个的用2个替代）
    text = re.sub(r'(\[EMOJI\]\s*){3,}', '[EMOJI] [EMOJI] ', text)
    
    return text


def chinese_specific_processing(text):
    """中文特定处理"""
    text = convert(text, 'zh-cn')  # 繁简转换
    
    # 标点符号处理
    text = text.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?')
    text = text.replace(',', '，').replace('.', '。').replace('!', '！').replace('?', '？')
    text = text.replace('…', '...').replace('——', '——')
    
    return text


def handle_special_characters(text):
    """处理特殊字符"""
    # URL替换
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
    # 邮箱替换
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # @提及替换
    text = re.sub(r'@\w+', '[MENTION]', text)
    # 匹配整数和浮点数（包括科学计数法）
    text = re.sub(r'[+-]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][+-]?\d+)?', '[NUM]', text)
    
    return text


def remove_kaomoji(text):
    """删除颜文字"""
    kaomoji_patterns = [
        r'[oO0~*^\-_][\._][oO0~*^\-_]',
        r'\([^)]*\)',
        r'[Tt][\._][Tt]',
        r'[xX][\._][xX]',
        r'>\.<', r'>_<',
        r'\^_\^', r'-_-', r'\._\.',
    ]
    
    for pattern in kaomoji_patterns:
        text = re.sub(pattern, '', text)
    
    return text


def filter_uncommon_chars(text):
    """过滤非常规字符"""
    # 定义允许的字符范围
    # 1. 中文字符范围
    chinese_chars = r'\u4e00-\u9fa5'
    
    # 2. 英文字母（大小写）
    english_chars = r'a-zA-Z'
    
    # 3. 数字
    digits = r'0-9'
    
    # 4. 常见中文标点符号
    chinese_punctuation = r'，。！？；："＂\'＇（）《》【】～…—'
    
    # 5. 常见英文标点符号
    english_punctuation = r',\.!?;:"\'\(\)\[\]{}@#%&\*\+\-=/\\\$\$€£'
    
    # 6. 空白字符（空格、换行、制表符等）
    whitespace = r'\s'
    
    # 组合所有允许的字符
    allowed_chars = f'[{chinese_chars}{english_chars}{digits}{chinese_punctuation}{english_punctuation}{whitespace}]'
    
    # 删除不在允许字符集中的所有字符
    text = re.sub(f'[^{allowed_chars[1:-1]}]', '', text)
    
    return text

def normalize_repetition(text):
    """规整重复字符"""
    text = re.sub(r'([!?。！？])\1{2,}', r'\1\1', text)  # 标点重复
    text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)  # 字母重复
    
    text = re.sub(r'\.{3}', '...', text)  # 确保...保持3个点
    text = re.sub(r'——', '——', text)     # 确保——保持两个长破折号
    
    return text


def final_cleaning(text):
    """最终清理"""
    text = re.sub(r' +', ' ', text)  # 移除多余空格
    text = text.strip()
    
    # 长度控制
    if len(text) > 400:
        text = text[:400] + '...'
    
    return text

def process_table_raw(engine):
    """
    从db读取raw表 → 处理数据 → 写入db的processed表
    """
    
    # 创建数据库引擎
    input_engine = engine
    output_engine = engine
    
    try:
        # 从原始表读取数据
        df = pd.read_sql_query("SELECT id, rating, review_content, likes, publish_time, game_name FROM raw", input_engine)
        print(f"成功读取 {len(df)} 条原始数据")
        
        # 复制DataFrame避免修改原始数据
        processed_df = df.copy()
        
        # 清洗review_content列
        if 'review_content' in processed_df.columns:
            processed_df['review_content'] = processed_df['review_content'].apply(clean_reviews)
            print(f"成功处理 {len(processed_df)} 条评论")
        
        
        # 写入目标表（自动创建processed表）
        try:
            processed_df.to_sql(
                'processed',
                output_engine,
                if_exists='replace',  # 替换现有表
                index=False,
                dtype={
                    'id': Integer,
                    'rating': Integer,
                    'review_content': Text,
                    'likes': Integer,
                    'publish_time': DateTime(),
                    'game_name': String,
                }
            )
            print("数据已成功写入 processed 表")
        except Exception as e:
            print(f"错误: {e}")
        
            
    finally:
        input_engine.dispose()


if __name__ == "__main__":
    # 将根目录添加到 sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    #  导入数据库模块 
    try:
        from app.database import Engine
        print("成功导入数据库模块")
    except ImportError as e:
        print(f"模块导入失败: {e}")
        sys.exit(1)
    process_table_raw(Engine)
    