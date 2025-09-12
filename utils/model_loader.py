import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

class BERTLocalLoader:
    """BERT本地模型加载和管理类"""
    
    def __init__(self, model_name: str = "bert-base-uncased", models_dir: str = "./models"):
        #设置模型路径./models/bert-base-uncased
        self.model_name = model_name
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, model_name)
        
        self.tokenizer = None
        self.model = None
        self.config = None
        
        # 检查模型文件
        self._check_model_files()
    
    def _check_model_files(self) -> None:
        """检查必要的模型文件是否存在"""
        required_files = [
            'config.json',
            'pytorch_model.bin', 
            'vocab.txt',
            'tokenizer.json',
            'tokenizer_config.json'
        ]
        
        for file in required_files:
            file_path = os.path.join(self.model_path, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"缺少模型文件: {file}")
    
    def load_tokenizer(self) -> AutoTokenizer:
        """加载分词器"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self.tokenizer
    
    def load_model(self, model_type: str = "base") -> AutoModel:
        """
        加载BERT模型
        
        Args:
            model_type: "base" - 基础模型, "cls" - 分类模型
        """
        if self.model is None:                        
            if model_type == "cls":
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            else:
                self.model = AutoModel.from_pretrained(self.model_path)  
        return self.model
    

# 单例模式，避免重复加载
bert_loader = BERTLocalLoader()