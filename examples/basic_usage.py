#!/usr/bin/env python3
"""
BERT本地模型基础使用示例
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import BERTLocalLoader, bert_loader
from utils.text_processor import TextProcessor

def demo_basic_loading():
    """演示基础加载"""
    print("🔧 1. 模型加载演示")
    print("=" * 50)
    
    # 初始化加载器
    loader = BERTLocalLoader()
    
    # 获取模型信息
    info = loader.get_model_info()
    print(f"📊 模型信息:")
    print(f"   - 参数量: {info['parameters']:,}")
    print(f"   - 隐藏层: {info['layers']}")
    print(f"   - 隐藏维度: {info['hidden_size']}")
    print(f"   - 注意力头: {info['attention_heads']}")
    print(f"   - 运行设备: {info['device']}")

def demo_text_processing():
    """演示文本处理"""
    print("\n📝 2. 文本处理演示")
    print("=" * 50)
    
    processor = TextProcessor()
    
    # 示例文本
    texts = [
        "Hello, BERT! This is amazing.",
        "I love natural language processing.",
        "Transformers are powerful deep learning models.",
        "The weather is beautiful today."
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. 处理文本: '{text}'")
        
        # 分词
        inputs = processor.tokenize_text(text)
        print(f"   Token IDs: {inputs['input_ids'].shape}")
        print(f"   注意力掩码: {inputs['attention_mask'].shape}")
        
        # 获取嵌入
        embeddings = processor.get_embeddings(text)
        print(f"   隐藏状态: {embeddings['last_hidden_state'].shape}")
        print(f"   句子嵌入: {embeddings['pooler_output'].shape}")

def demo_advanced_features():
    """演示高级功能"""
    print("\n🎯 3. 高级功能演示")
    print("=" * 50)
    
    processor = TextProcessor()
    
    # 批量处理
    texts = ["Text " + str(i) for i in range(10)]
    batch_embeddings = processor.batch_process(texts, batch_size=4)
    print(f"📦 批量处理 {len(texts)} 个文本")
    print(f"   批量嵌入形状: {batch_embeddings.shape}")
    
    # 单个句子嵌入
    sample_text = "This is a sample sentence for embedding."
    sentence_embedding = processor.get_sentence_embedding(sample_text)
    print(f"\n🔍 句子嵌入示例:")
    print(f"   文本: '{sample_text}'")
    print(f"   嵌入维度: {sentence_embedding.shape}")
    print(f"   前5个值: {sentence_embedding[:5].cpu().numpy()}")

if __name__ == "__main__":
    print("🤖 BERT本地模型完全使用指南")
    print("📍 模型路径: ./models/bert-base-uncased/")
    print("-" * 60)
    
    try:
        # 运行演示
        demo_basic_loading()
        demo_text_processing()
        demo_advanced_features()
        
        print("\n" + "=" * 60)
        print("🎉 所有演示完成！您现在可以：")
        print("   1. 修改文本进行实验")
        print("   2. 尝试不同的批量大小")
        print("   3. 探索其他BERT功能")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查模型文件是否完整下载")