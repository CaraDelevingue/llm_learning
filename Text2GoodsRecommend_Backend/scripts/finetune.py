import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(DATASET_DIR):
    """
    加载和预处理数据
    """
    print("加载预处理数据...")
    #数据集所在文件夹目录
    #加载预处理后的数据和标签编码器
    train_df = pd.read_csv(os.path.join(DATASET_DIR,'train.csv'))
    val_df = pd.read_csv(os.path.join(DATASET_DIR,'validation.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_DIR,'test.csv'))

    label_encoder = joblib.load(os.path.join(DATASET_DIR,'label_encoder.pkl'))
    num_labels = len(label_encoder.classes_)

    '''
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")
    print(f"游戏类别数: {num_labels}")
    '''
    return train_df, val_df, test_df, label_encoder,num_labels


def setup_model_and_tokenizer(model_path, num_labels, label_encoder):
    """
    设置模型和分词器
    """
    print(f"加载模型和分词器: {model_path}")
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )
    
    print("模型和分词器加载完成！")
    return tokenizer, model



def prepare_datasets(train_df, val_df, test_df, tokenizer):
    """
    准备训练数据集
    """
    print("准备数据集...")
    # 数据预处理和转换为模型输入格式
    print("预处理数据...")

    # 将DataFrame转换为Hugging Face Dataset格式
    train_dataset = Dataset.from_dict({
        "text": train_df["review_content"].tolist(),
        "label": train_df["label"].tolist()
    })
    val_dataset = Dataset.from_dict({
        "text": val_df["review_content"].tolist(),
        "label": val_df["label"].tolist()
    })
    test_dataset = Dataset.from_dict({
        "text": test_df["review_content"].tolist(),
        "label": test_df["label"].tolist()
    })

    # 定义tokenize函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding=True, 
            max_length=128,
            return_tensors=None
        )

    # 应用tokenize
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    print("数据预处理完成！")
    return tokenized_train, tokenized_val, tokenized_test


# 定义评估指标和创建Trainer
def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "f1_weighted": f1_score(labels, predictions, average='weighted')
    }

def setup_training_args(OUTPUT_DIR):
    """
    设置训练参数
    """
    # 根据本地硬件调整参数
    # 如果使用CPU，考虑减小批次大小和序列长度
    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        
        # 训练参数 - 根据本地硬件调整
        num_train_epochs=3,
        per_device_train_batch_size=4 if not use_cuda else 8,  # CPU上使用较小的批次
        per_device_eval_batch_size=4 if not use_cuda else 8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        
        # 评估与保存策略
        eval_strategy="epoch",           
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_weighted",
        
        # 日志与报告
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",  # 本地运行不需要外部报告
        
        # 优化设置
        fp16=use_cuda,  # 仅在GPU上使用混合精度
        dataloader_num_workers=0 if not use_cuda else 2,  # CPU上不使用多线程
    )
    
    print("训练参数设置完成！")
    return training_args


def train_model(model, training_args, tokenized_train, tokenized_val, tokenizer, label_encoder,SAVE_MODEL_DIR):
    """
    训练模型
    """
    # 创建数据收集器（用于动态padding）
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Trainer创建完成！")

    print("开始模型训练...")

    # 训练模型
    train_results = trainer.train()

    # 保存最终模型
    trainer.save_model(SAVE_MODEL_DIR)
    tokenizer.save_pretrained(SAVE_MODEL_DIR)

    # 保存标签编码器
    joblib.dump(label_encoder, os.path.join(SAVE_MODEL_DIR,'label_encoder.pkl'))

    print("训练完成！最终模型已保存。")
    return trainer

def evaluate_model(trainer, tokenized_test,SAVE_MODEL_DIR):
    """
    评估模型
    """
    print("在测试集上评估模型...")

    # 评估测试集
    test_results = trainer.evaluate(tokenized_test)
    print("\n测试集评估结果:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")

    # 保存评估结果
    with open(os.path.join(SAVE_MODEL_DIR,'test_results.txt'), "w") as f:
        for key, value in test_results.items():
            f.write(f"{key}: {value:.4f}\n")
    return test_results


def predict_game(review_content, rating, likes, SAVE_MODEL_DIR):
        """使用训练好的模型预测游戏名称"""
        # 加载模型和组件
        model_path=SAVE_MODEL_DIR
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")
        
        # 准备输入（与训练时相同的格式）
        enhanced_text = f"评分:{rating}星 点赞数:{likes} {review_content}"
        
        # 预测
        inputs = tokenizer(enhanced_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            predicted_game = label_encoder.inverse_transform([predicted_class])[0]
            confidence = predictions[0][predicted_class].item()
        
        return predicted_game, confidence
    
    
def predict_example(SAVE_MODEL_DIR):
    """
    预测示例
    """
    # 测试预测函数
    print("\n预测示例:")
    sample_review = "这个游戏画面精美，操作流畅，剧情引人入胜"
    sample_rating = 5
    sample_likes = 150

    predicted_game, confidence = predict_game(sample_review, sample_rating, sample_likes,SAVE_MODEL_DIR)
    print(f"评论: {sample_review}")
    print(f"预测游戏: {predicted_game}")
    print(f"置信度: {confidence:.4f}")


def main():
    """
    主函数
    """
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(ROOT_DIR,'data','processed')
    OUTPUT_DIR = os.path.join(ROOT_DIR,'models','game_classification_output')
    SAVE_MODEL_DIR = os.path.join(ROOT_DIR,'models','game_classification_model')
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载和预处理数据
    train_df, val_df, test_df, label_encoder,num_labels = load_and_preprocess_data(DATASET_DIR)
    
    # 2. 设置模型和分词器
    # 替换为你的本地模型路径
    model_path = os.path.join(ROOT_DIR,'models','chinese_roberta_wwm_ext')  # 本地模型路径
    
    tokenizer, model = setup_model_and_tokenizer(model_path, num_labels, label_encoder)
    
    # 3. 准备数据集
    tokenized_train, tokenized_val, tokenized_test = prepare_datasets(train_df, val_df, test_df, tokenizer)
    
    # 4. 设置训练参数
    training_args = setup_training_args(OUTPUT_DIR)
    
    # 5. 训练模型
    trainer = train_model(model, training_args, tokenized_train, tokenized_val, tokenizer, label_encoder,SAVE_MODEL_DIR)
    
    # 6. 评估模型
    test_results = evaluate_model(trainer, tokenized_test,SAVE_MODEL_DIR)
    
    # 7. 预测示例
    predict_example(SAVE_MODEL_DIR)

if __name__ == "__main__":
    main()
