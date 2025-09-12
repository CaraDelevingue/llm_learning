import torch
import torch.nn as nn
from utils.text_processor import TextProcessor

class SimpleTextClassifier(nn.Module):
    """基于BERT的简单文本分类器"""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.text_processor = TextProcessor()
        self.classifier = nn.Linear(768, num_classes)  # BERT隐藏维度是768
        
    def forward(self, texts):
        # 获取BERT嵌入
        with torch.no_grad():  # 冻结BERT权重
            embeddings = self.text_processor.get_embeddings(texts)
            cls_embeddings = embeddings['pooler_output']
        
        # 分类
        return self.classifier(cls_embeddings)

# 使用示例
if __name__ == "__main__":
    print("🧠 基于BERT的文本分类示例")
    
    # 初始化分类器
    classifier = SimpleTextClassifier(num_classes=3)
    
    # 示例文本和标签（情感分析）
    texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, not great.",
        "Amazing experience!",
        "Very disappointed."
    ]
    
    # 假想的预测
    with torch.no_grad():
        predictions = classifier(texts)
        probs = torch.softmax(predictions, dim=1)
        
        print("\n📊 分类预测:")
        for i, (text, prob) in enumerate(zip(texts, probs)):
            print(f"{i+1}. '{text}'")
            print(f"   概率: {prob.cpu().numpy().round(3)}")