# model_loader.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(ROOT_DIR,"models","mt5-xlsum")

class SummaryModelLoader:
    """摘要模型加载器"""
     
    def __init__(self,model_path=None):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        
    def _get_device(self):
        """自动选择设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_model(self):
        """加载模型和分词器"""
        try:
            logger.info(f"开始加载模型从: {self.model_path}")
            logger.info(f"使用设备: {self.device}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("分词器加载成功")
            # 加载模型
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            logger.info("✅ 模型加载成功!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def is_loaded(self):
        """检查模型是否已加载"""
        return self.tokenizer is not None and self.model is not None
    
    def get_model_info(self):
        """获取模型信息"""
        if not self.is_loaded():
            return {"error": "模型未加载"}
        
        return {
            "model_type": type(self.model).__name__,
            "tokenizer_type": type(self.tokenizer).__name__,
            "device": str(self.device),
            "model_path": self.model_path,
            "vocab_size": len(self.tokenizer)
        }


class SummarizationService:
    """摘要服务"""
    
    def __init__(self, model_loader: SummaryModelLoader):
        self.loader = model_loader
        if not self.loader.is_loaded():
            raise ValueError("模型未加载，请先调用 load_model()")
    
    def summarize(self, text: str, max_length: int = None, min_length: int = None, 
                  num_beams: int = None, do_sample: bool = False) -> dict:
        """
        文本摘要服务
        
        Args:
            text: 输入文本
            max_length: 摘要最大长度
            min_length: 摘要最小长度
            num_beams: beam search 参数
            do_sample: 是否采样
            
        Returns:
            dict: 包含摘要和元数据的结果
        """
        try:
            # 输入验证
            if not text or not text.strip():
                return {
                    "success": False,
                    "error": "输入文本不能为空"
                }
            
            original_text = text.strip()
            original_length = len(original_text)

            
            # 根据文本长度自适应调整参数
            if original_length < 100:
                # 短文本参数 - 严格控制，避免幻觉
                adaptive_max_length = min(40, original_length // 2)
                adaptive_min_length = max(10, original_length // 4)
                adaptive_num_beams = 2  # 减少beam数，避免过度生成
                temperature = 0.3  # 低温度，减少随机性
                repetition_penalty = 2.0  # 高重复惩罚
                no_repeat_ngram_size=2
            elif original_length < 300:
                # 中等长度文本
                adaptive_max_length = min(80, original_length // 3)
                adaptive_min_length = max(20, original_length // 6)
                adaptive_num_beams = 3
                temperature = 0.5
                repetition_penalty = 1.5
                no_repeat_ngram_size=3
            else:
                # 长文本
                adaptive_max_length = min(150, original_length // 4)
                adaptive_min_length = max(30, original_length // 8)
                adaptive_num_beams = 4
                temperature = 0.7
                repetition_penalty = 1.2
                no_repeat_ngram_size=3
                
            # 添加明确的指令提示词
            if original_length < 100:
                # 短文本使用更强的指令
                prompt = f"请用一句话简要总结: {original_text}"
            else:
                prompt = f"请总结以下内容: {original_text}"
                
                
            # 文本编码
            inputs = self.loader.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                padding="longest",
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.loader.device) for k, v in inputs.items()}
            
            # 生成摘要
            with torch.no_grad():
                summary_ids = self.loader.model.generate(
                    inputs["input_ids"],
                    max_length=adaptive_max_length,
                    min_length=adaptive_min_length,
                    num_beams=adaptive_num_beams,
                    do_sample=False,
                    early_stopping=True,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    repetition_penalty=repetition_penalty,  
                    #长度惩罚 ： False
                    length_penalty=1.0,  
                    temperature=temperature,
                )
            
            # 解码摘要
            summary_text = self.loader.tokenizer.decode(
                summary_ids[0], 
                skip_special_tokens=True
            )
            
            # 后处理：移除可能的提示词残留
            summary_text = summary_text.replace("请用一句话简要总结:", "").replace("请总结以下内容:", "").strip()
            
            # 质量检查：如果摘要明显比原文长或完全无关，返回原文前部分
            if (len(summary_text) > original_length * 0.8 or 
                not any(word in summary_text for word in original_text[:20])):
                # 返回原文的精简版本
                summary_text = original_text[:adaptive_max_length] + "..."
            
            
            summary_length = len(summary_text)
            compression_ratio = 1 - (summary_length / original_length) if original_length > 0 else 0
            
            return {
                "success": True,
                "summary": summary_text,
                "original_length": original_length,
                "summary_length": summary_length,
                "compression_ratio": round(compression_ratio, 4),
                "model_used": "bart-base-chinese"
            }
            
        except Exception as e:
            logger.error(f"摘要生成失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def batch_summarize(self, texts: list, **kwargs) -> list:
        """批量摘要"""
        results = []
        for text in texts:
            result = self.summarize(text, **kwargs)
            results.append(result)
        return results


# 全局服务实例
_model_loader = None
_summarization_service = None

def get_summarization_service(model_path: str=model_path ) -> SummarizationService:
    """获取摘要服务实例（单例模式）"""
    global _model_loader, _summarization_service
    
    if _summarization_service is None:
        _model_loader = SummaryModelLoader(model_path)
        if _model_loader.load_model():
            _summarization_service = SummarizationService(_model_loader)
        else:
            raise RuntimeError("无法加载模型")
    
    return _summarization_service