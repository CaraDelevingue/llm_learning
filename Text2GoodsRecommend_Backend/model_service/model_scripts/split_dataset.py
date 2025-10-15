from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import sys
import os
from sqlalchemy import  Integer, Float, Text,String
from sqlalchemy import text

def triple_stratified_split(df, train_ratio = 0.80, val_ratio=0.10, test_ratio=0.10):
    """
    三级分层抽样：游戏名 -> 高赞/低赞 -> 星级 -> 时间顺序
    """
    # 检查比例是否正确
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 复制数据避免修改原数据
    df = df.copy()
    
    # 保存原始列名
    original_columns = df.columns.tolist()
    
    # 第一步：按游戏名分组，在每个游戏内确定高赞阈值（前30%）
    df['high_likes'] = False
    for game in df['game_name'].unique():
        game_mask = df['game_name'] == game
        game_likes = df.loc[game_mask, 'likes']
        if len(game_likes) > 0:
            threshold = game_likes.quantile(0.7)  # 前30%为高赞
            df.loc[game_mask, 'high_likes'] = df.loc[game_mask, 'likes'] >= threshold
    
    # 创建分层标签：游戏名_高赞_星级
    df['strata'] = (df['game_name'].astype(str) + '_' + 
                   df['high_likes'].astype(str) + '_' + 
                   df['rating'].astype(str))
    
    # 按时间排序
    time_column = 'publish_time' 
    df = df.sort_values(by=[time_column], ascending=True)  # 先发布的在前面
    
    # 存储最终结果
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # 对每个分层单独处理
    for stratum in df['strata'].unique():
        stratum_data = df[df['strata'] == stratum].copy()
        
        if len(stratum_data) == 0:
            continue
            
        # 计算每个分层中应该分配到各集合的数量
        n_total = len(stratum_data)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, int(n_total * test_ratio))
        
        # 调整数量以确保总和等于n_total
        total_allocated = n_train + n_val + n_test
        if total_allocated < n_total:
            n_train += (n_total - total_allocated)
        elif total_allocated > n_total:
            n_train -= (total_allocated - n_total)
        
        # 按时间顺序分配（先发布的优先进入训练集）
        indices = list(range(len(stratum_data)))
        
        # 训练集：最早发布的
        train_indices = indices[:n_train]
        # 验证集：中间发布的
        val_indices = indices[n_train:n_train + n_val]
        # 测试集：最新发布的
        test_indices = indices[n_train + n_val:n_train + n_val + n_test]
        
        # 添加到对应的集合
        train_dfs.append(stratum_data.iloc[train_indices])
        val_dfs.append(stratum_data.iloc[val_indices])
        test_dfs.append(stratum_data.iloc[test_indices])
    
    # 合并所有分层的结果
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # 删除新增的临时列，只保留原始列
    columns_to_drop = ['high_likes', 'strata']
    train_df = train_df.drop(columns=columns_to_drop)
    val_df = val_df.drop(columns=columns_to_drop)
    test_df = test_df.drop(columns=columns_to_drop)
    
    # 确保列顺序与原始数据一致
    train_df = train_df[original_columns]
    val_df = val_df[original_columns]
    test_df = test_df[original_columns]
    
    # 打印详细的分布信息
    #print_distribution_info(df, train_df, val_df, test_df)
    
    return train_df, val_df, test_df

    
def print_distribution_info(original_df, train_df, val_df, test_df):
    """打印详细的数据分布信息"""
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"总计: {len(train_df) + len(val_df) + len(test_df)}")
    
    print("\n=== 按游戏分布 ===")
    for game in original_df['game_name'].unique():
        total = len(original_df[original_df['game_name'] == game])
        train_count = len(train_df[train_df['game_name'] == game])
        val_count = len(val_df[val_df['game_name'] == game])
        test_count = len(test_df[test_df['game_name'] == game])
        
        print(f"{game}: 总数{total} -> 训练{train_count}({train_count/total:.1%}) "
              f"验证{val_count}({val_count/total:.1%}) 测试{test_count}({test_count/total:.1%})")
    
    print("\n=== 按高赞/低赞分布 ===")
    for likes_type in [True, False]:
        type_name = "高赞" if likes_type else "低赞"
        total = len(original_df[original_df['high_likes'] == likes_type])
        train_count = len(train_df[train_df['high_likes'] == likes_type])
        val_count = len(val_df[val_df['high_likes'] == likes_type])
        test_count = len(test_df[test_df['high_likes'] == likes_type])
        
        if total > 0:
            print(f"{type_name}: 总数{total} -> 训练{train_count}({train_count/total:.1%}) "
                  f"验证{val_count}({val_count/total:.1%}) 测试{test_count}({test_count/total:.1%})")
    
    print("\n=== 按星级分布 ===")
    for rating in sorted(original_df['rating'].unique()):
        total = len(original_df[original_df['rating'] == rating])
        train_count = len(train_df[train_df['rating'] == rating])
        val_count = len(val_df[val_df['rating'] == rating])
        test_count = len(test_df[test_df['rating'] == rating])
        
        print(f"{rating}星: 总数{total} -> 训练{train_count}({train_count/total:.1%}) "
              f"验证{val_count}({val_count/total:.1%}) 测试{test_count}({test_count/total:.1%})")
    
 

def save_datasets_to_db(train_df, val_df, test_df, engine):
    """
    将训练集、验证集和测试集保存到数据库中的三个表
    """
    
    # 定义要保存的列
    columns_to_save = ['id', 'rating', 'review_content', 'likes', 'game_name']
    try:
        # 保存训练集到train表
        train_df[columns_to_save].to_sql(
            'train', 
            engine, 
            if_exists='replace', 
            index=False,
            dtype={
                    'id': Integer,
                    'rating': Float,
                    'review_content': Text,
                    'likes': Integer,
                    'game_name': String,
                }
        )
        print("训练集已保存到 'train' 表")
        
        # 保存验证集到validation表
        val_df[columns_to_save].to_sql(
            'validation', 
            engine, 
            if_exists='replace', 
            index=False,
            dtype={
                    'id': Integer,
                    'rating': Float,
                    'review_content': Text,
                    'likes': Integer,
                    'game_name': String,
                }
        )
        print("验证集已保存到 'validation' 表")
        
        # 保存测试集到test表
        test_df[columns_to_save].to_sql(
            'test', 
            engine, 
            if_exists='replace', 
            index=False,
            dtype={
                    'id': Integer,
                    'rating': Float,
                    'review_content': Text,
                    'likes': Integer,
                    'game_name': String,
                }
        )
        print("测试集已保存到 'test' 表")
        
        # 打印各表的大小信息
        print(f"\n各表记录数:")
        print(f"train表: {len(train_df)} 条记录")
        print(f"validation表: {len(val_df)} 条记录")
        print(f"test表: {len(test_df)} 条记录")
    except Exception as e:
        print(f"保存数据到数据库时出错: {e}")



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
        
    try:
        # 从原始表读取数据
        processed_df = pd.read_sql_query("SELECT id, rating, review_content, likes, publish_time, game_name FROM processed", Engine)
        print(f"成功读取 {len(processed_df)} 条原始数据")
    except Exception as e:
            print(f"从processed表读取数据错误: {e}")


    # 分割数据集
    train_df, val_df, test_df = triple_stratified_split(processed_df)
    # 将分割的数据集分别存入相应的表中
    save_datasets_to_db(train_df, val_df, test_df, Engine)