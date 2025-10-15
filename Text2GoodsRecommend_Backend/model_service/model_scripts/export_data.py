import pandas as pd
import sys,os


# 将根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 指定输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'model_service','data','processed')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"输出目录已创建: {OUTPUT_DIR}")

#  导入数据库模块 
try:
    from app.database import Engine
    print("成功导入数据库模块")
except ImportError as e:
    print(f" 模块导入失败: {e}")
    sys.exit(1)


# 创建连接
try:
    tables = ['train', 'validation', 'test']
    
    for table_name in tables:
        # 读取数据
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", Engine)
        
        # 生成输出文件路径
        output_file = os.path.join(OUTPUT_DIR, f"{table_name}.csv")
        
        # 导出到 CSV
        df.to_csv(output_file, index=False, encoding='gbk')
        # 验证导出
        print(f"成功导出到: {output_file}")
        print(f"数据行数: {len(df)}")
        #print(f"前5行预览:")
        #print(df.head().to_markdown(index=False))
        
    print("\n所有表导出完成！")
    print(f"CSV 文件已保存到: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"导出过程中发生错误: {str(e)}")
    # 详细错误信息（用于调试）
    import traceback
    traceback.print_exc()

finally:
    # 关闭数据库连接
    if 'engine' in locals():
        Engine.dispose()
    print("数据库连接已关闭")