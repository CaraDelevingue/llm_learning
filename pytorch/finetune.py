import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import time
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 数据加载器
class ChineseReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.texts = df['review_content'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Tokenization
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 2. 模型定义（针对Chinese-RoBERTa-wwm-ext）
class ChineseTextClassifier(nn.Module):
    def __init__(self, model_path, num_classes, use_auxiliary=True):
        super().__init__()
        # 加载本地Chinese-RoBERTa模型
        self.roberta = AutoModel.from_pretrained(model_path)
        self.config = self.roberta.config
        self.num_classes = num_classes
        
        # 分类器
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # 文本特征提取
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.pooler_output  # [CLS] token的特征
        
        # 分类
        logits = self.classifier(text_features)
        return logits

# 3. 训练函数
def train_model(model, train_loader, val_loader, epochs, device, model_save_path):
    optimizer = AdamW([
        {'params': model.roberta.parameters(), 'lr': 2e-5},
        {'params': [p for n, p in model.named_parameters() if 'roberta' not in n], 'lr': 1e-3}
    ])
    
    # 学习率调度器
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    
    # 训练历史记录
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # 训练阶段
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc="训练中")
        for batch in progress_bar:
            # 移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播 - 只使用文本
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_correct/train_total:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(avg_loss)
        
        # 验证阶段
        val_acc = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)
        
        print(f"训练损失: {avg_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{model_save_path}/best_model.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), f'{model_save_path}/final_model.pth')
    print(f"训练完成，最终模型已保存到 {model_save_path}")
    
    return train_losses, val_accuracies

def evaluate_model(model, data_loader, device):
    """使用文本-only模式评估"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="验证中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 不使用rating和likes
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy

# 4. 测试函数
def test_model(model, test_loader, device, label_encoder):
    """测试模型性能"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 使用文本-only模式（模拟部署）
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    # 将类别名称转换为字符串
    class_names = [str(cls) for cls in label_encoder.classes_]
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(report)
    
    return accuracy, all_predictions, all_labels

# 5. 部署推理类
class DeploymentPredictor:
    def __init__(self, model_path, tokenizer_path, label_encoder_path, num_classes, device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = ChineseTextClassifier(model_path, num_classes)
        self.model.load_state_dict(torch.load(f'{model_path}/best_model.pth', map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.label_encoder = joblib.load(label_encoder_path)
        self.device = device
    
    def predict(self, text):
        """部署预测：只输入文本"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(outputs, dim=1).cpu().item()
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'probabilities': probabilities.cpu().numpy()[0]
        }

# 6. 主训练流程
def main():
    # 配置参数
    # 设置文件根目录
    ROOT_DIR =os.path.dirname(os.path.abspath(__file__))
    # 数据库目录
    DATASET_DIR = os.path.join(ROOT_DIR,'data')
    # 本地模型路径  
    MODEL_PATH = os.path.join(ROOT_DIR,'model','chinese_roberta_wwm_ext')
    # 模型保存地址
    SAVE_MODEL_DIR = os.path.join(ROOT_DIR,'trained_model')
    # 数据保存地址
    SAVE_OUTPUT_DIR = os.path.join(ROOT_DIR,'train_output')
    
    
    BATCH_SIZE = 16
    EPOCHS = 5
    MAX_LENGTH = 256
    
    
    # 加载预处理后的数据
    print("加载预处理数据...")
    train_df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATASET_DIR, 'validation.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
    
    # 加载标签编码器
    label_encoder = joblib.load(os.path.join(DATASET_DIR, 'label_encoder.pkl'))
    num_classes = len(label_encoder.classes_)
    
    print(f"数据加载完成:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条") 
    print(f"  测试集: {len(test_df)} 条")
    print(f"  类别数: {num_classes}")
    print(f"  类别: {list(label_encoder.classes_)}")
    
    # 初始化tokenizer和模型
    print("初始化模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = ChineseTextClassifier(MODEL_PATH, num_classes)
    model.to(device)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_dataset = ChineseReviewDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = ChineseReviewDataset(val_df, tokenizer, MAX_LENGTH)
    test_dataset = ChineseReviewDataset(test_df, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 训练模型
    print("开始训练...")
    start_time = time.time()
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, EPOCHS, device, SAVE_MODEL_DIR
    )
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")
    
    # 测试模型
    print("\n测试模型性能...")
    test_accuracy, test_predictions, test_labels = test_model(
        model, test_loader, device, label_encoder
    )
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy,
        'training_time': training_time
    }
    
    joblib.dump(history, os.path.join(SAVE_OUTPUT_DIR, 'training_history.pkl'))
    
    # 保存tokenizer和标签编码器
    tokenizer.save_pretrained(SAVE_MODEL_DIR)
    joblib.dump(label_encoder, os.path.join(SAVE_MODEL_DIR, 'label_encoder.pkl'))
    
    print(f"\n所有文件已保存到 {SAVE_MODEL_DIR} 目录")
    print("训练完成！")

if __name__ == "__main__":
    main()