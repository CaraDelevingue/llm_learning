import torch
from typing import List, Dict, Union
from .model_loader import bert_loader

class TextProcessor:
    """文本处理工具类"""
    
    def __init__(self):
        self.tokenizer = bert_loader.load_tokenizer()
        self.model = bert_loader.load_model()
    
    def tokenize_text(self, text: str, return_tensors: str = "pt") -> Dict:
        """
        文本分词
        
        Args:
            text: 输入文本
            return_tensors: 返回张量类型 ("pt" for PyTorch)
        """
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            truncation=True,
            padding=True,
            max_length=512,
            add_special_tokens=True
        )
    
    def get_embeddings(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        获取文本嵌入向量
        
        Args:
            text: 单个文本或文本列表
            
        Returns:
            last_hidden_state: 最后一层隐藏状态
            pooler_output: 池化后的输出 [CLS] token
        """
        if isinstance(text, str):
            text = [text]
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # 移动到GPU（如果可用）
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return {
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'attention_mask': inputs['attention_mask']
        }
    
    def get_sentence_embedding(self, text: str) -> torch.Tensor:
        """
        获取句子级别的嵌入（使用[CLS] token）
        """
        embeddings = self.get_embeddings(text)
        return embeddings['pooler_output'][0]  # 取第一个句子的[CLS]嵌入
    
    def batch_process(self, texts: List[str], batch_size: int = 8) -> List[torch.Tensor]:
        """
        批量处理文本
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.get_embeddings(batch_texts)
            all_embeddings.append(embeddings['pooler_output'])
        
        return torch.cat(all_embeddings, dim=0)