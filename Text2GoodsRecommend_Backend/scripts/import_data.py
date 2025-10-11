"""
数据导入脚本：将 CSV 数据导入数据库
"""

import sys
import os
import pandas as pd
from sqlalchemy.orm import Session

#  添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#  导入数据库模块 
try:
    from app.database.connection import engine, SessionLocal
    from app.database.models import Raw  # 确保模型已定义
    print("成功导入数据库模块")
except ImportError as e:
    print(f" 模块导入失败: {e}")
    sys.exit(1)

# === 数据导入函数 ===
def import_csv_data(csv_file_path: str):
    """
    从 CSV 文件导入数据到数据库
    """
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"CSV 文件不存在: {csv_file_path}")
        return False

    # 创建数据库会话
    db: Session = SessionLocal()
    
    try:
        print(f" 正在读取 CSV 文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path, encoding='gbk')

        # --- 数据验证 ---
        required_columns = ['rating', 'review_content', 'likes', 'publish_time','device_model','game_name']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV 缺少必要列: {missing_cols}")

        print(f"CSV 文件包含 {len(df)} 行，{len(df.columns)} 列")

        # --- 数据清洗 ---
        print("开始数据清洗...")
        # 去除关键字段为空的行
        initial_count = len(df)
        df = df.dropna(subset=['rating', 'review_content','publish_time','game_name'])
        cleaned_count = len(df)
        print(f"清洗后保留 {cleaned_count} 行（删除 {initial_count - cleaned_count} 行空数据）")

        # 填充默认值
        df['likes'] = df['likes'].fillna(0).astype(int)
        df['device_model'] = df['device_model'].fillna('Unknown')

        # --- 数据插入 ---
        print("开始插入数据库...")
        data = df[required_columns].to_dict(orient='records')
        
        db.bulk_insert_mappings(Raw, data)
        db.commit()
        
        print(f"成功导入 {len(data)} 条数据到数据库！")
        return True

    except Exception as e:
        db.rollback()
        print(f"数据导入失败: {e}")
        return False

    finally:
        db.close()
        print("数据库连接已关闭")

# === 4. 主程序 ===
if __name__ == "__main__":
    # 构造 CSV 文件路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "raw", "taptap_game_reviews.csv")
    
    print(f" CSV 路径: {csv_path}")
    
    # 运行导入
    import_csv_data(csv_path)