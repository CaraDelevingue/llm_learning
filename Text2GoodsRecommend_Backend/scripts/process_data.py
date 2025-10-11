import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


from sklearn.preprocessing import LabelEncoder

def encode_labels(df, label_encoder=None):
    """
    将游戏按照类型大类编码为数字标签
    """
    # 定义游戏分类映射
    game_categories = {
        # 二次元角色扮演游戏 (7款) - 标签 0
        0: ['原神', '崩坏3', '崩坏星穹铁道', '命运冠位指定', '明日方舟', '战双帕弥什', '碧蓝航线'],
        
        # 传统角色扮演游戏 (6款) - 标签 1
        1: ['深空之眼', '蔚蓝档案', '重返未来1999', '阴阳师', '逆水寒', '鸣潮'],
        
        # 射击游戏 (5款) - 标签 2
        2: ['三角洲部队', '和平精英', '香肠派对', '尘白禁区', '绝区零'],
        
        # 动作游戏 (5款) - 标签 3
        3: ['永劫无间', '死亡细胞', '艾希', '元气骑士', '火影忍者'],
        
        # 音乐游戏 (4款) - 标签 4
        4: ['喵斯快跑', '音乐世界赛特斯2', '菲格罗斯', '梦想协奏曲！少女乐团派对！'],
        
        # 恋爱模拟游戏 (3款) - 标签 5
        5: ['光与夜之恋', '恋与深空', '未定事件簿'],
        
        # 沙盒创造游戏 (3款) - 标签 6
        6: ['我的世界安卓版', '泰拉瑞亚', '光遇'],
        
        # MOBA与竞技游戏 (4款) - 标签 7
        7: ['王者荣耀', '金铲铲之战', '第五人格', '燕云十六声'],
        
        # 休闲游戏 (3款) - 标签 8
        8: ['蛋仔派对', '我在七年后等着你', '无限暖暖']
    }
    
    # 创建游戏到数字标签的映射
    game_to_label = {}
    for label, games in game_categories.items():
        for game in games:
            game_to_label[game] = label
    
    # 直接添加数字标签，不创建category列
    df['label'] = df['game_name'].map(game_to_label)
    
    # 创建LabelEncoder用于后续使用（如果需要）
    if label_encoder is None:
        label_encoder = LabelEncoder()
        # 为了保持兼容性，我们拟合所有可能的标签
        label_encoder.fit(list(game_categories.keys()))
    
    return df, label_encoder

def clean_data(df):
    """
    数据清洗：处理缺失值和异常值
    """
    # 检查并处理缺失值
    if df.isnull().sum().sum() > 0:
        #print(f"发现缺失值: {df.isnull().sum()}")
        # 删除包含缺失值的行
        df = df.dropna()
        #print(f"删除缺失值后数据大小: {len(df)}")
    
    # 确保rating在合理范围内（假设1-5）
    if 'rating' in df.columns:
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
    
    # 确保likes非负
    if 'likes' in df.columns:
        df = df[df['likes'] >= 0]
    
    return df

 
def main():
    #数据集所在文件夹目录
    DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data','processed')
    #从分类的数据集中加载数据
    train_df = pd.read_csv(os.path.join(DATASET_DIR,'train.csv'),encoding='gbk')
    val_df = pd.read_csv(os.path.join(DATASET_DIR,'validation.csv'),encoding='gbk')  
    test_df = pd.read_csv(os.path.join(DATASET_DIR,'test.csv'),encoding='gbk')
    
    '''
    print("数据集加载完成！")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    
    # 检查数据列和基本信息
    print("训练集列名:", train_df.columns.tolist())
    print("\n训练集前5行:")
    print(train_df.head())

    print("\n训练集基本信息:")
    print(train_df.info())

    print("\n训练集数值列统计:")
    print(train_df.describe())

    # 检查缺失值
    print("\n训练集缺失值统计:")
    print(train_df.isnull().sum())

    # 检查game_name的分布
    print("\n游戏名称分布:")
    print(train_df['game_name'].value_counts())
    '''
        
    # 数据清洗
    #print("清洗数据...")
    train_df = clean_data(train_df)
    val_df = clean_data(val_df)
    test_df = clean_data(test_df)

    #print(f"清洗后数据大小 - 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")


    # 编码标签
    #print("编码标签...")
    train_df, label_encoder = encode_labels(train_df)
    val_df, _ = encode_labels(val_df, label_encoder)
    test_df, _ = encode_labels(test_df, label_encoder)

    '''
    print(f"游戏类别数量: {len(label_encoder.classes_)}")
    print("游戏名称映射:")
    for i, game in enumerate(label_encoder.classes_):
        print(f"  {i}: {game}")
    '''

    # 保存预处理后的数据
    train_df.to_csv(os.path.join(DATASET_DIR,'train.csv'), index=False)
    val_df.to_csv(os.path.join(DATASET_DIR,'validation.csv'), index=False)
    test_df.to_csv(os.path.join(DATASET_DIR,'test.csv'), index=False)

    # 保存标签编码器
    joblib.dump(label_encoder, os.path.join(DATASET_DIR,'label_encoder.pkl'))


    '''
    print(f"预处理后的数据已保存到 {DATASET_DIR}目录")
    print("文件列表:")
    for file in os.listdir(DATASET_DIR):
        print(f"  {file}")
    ''' 
    # 验证保存的数据可以正确加载
    print("验证保存的数据...")

    # 加载保存的数据
    train_loaded = pd.read_csv(os.path.join(DATASET_DIR,'train.csv'))
    val_loaded = pd.read_csv(os.path.join(DATASET_DIR,'validation.csv'))
    test_loaded = pd.read_csv(os.path.join(DATASET_DIR,'test.csv'))

    # 加载标签编码器
    label_encoder_loaded = joblib.load(os.path.join(DATASET_DIR,'label_encoder.pkl'))

    print(f"加载的数据大小 - 训练集: {len(train_loaded)}, 验证集: {len(val_loaded)}, 测试集: {len(test_loaded)}")
    print("标签编码器类别:", label_encoder_loaded.classes_)

    # 检查数据完整性
    print("\n数据完整性检查:")
    print("训练集列名:", train_loaded.columns.tolist())
        

    print("最终数据格式:")
    print("训练集:", train_loaded.shape)
    print("验证集:", val_loaded.shape)
    print("测试集:", test_loaded.shape)


    print("最终格式的数据已保存")
    
    

if __name__ == "__main__":
    main()
