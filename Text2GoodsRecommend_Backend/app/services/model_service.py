import sys,os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model_service.model_loader import get_summarization_service


def model_service(text: str):
    try:
        # 获取服务实例
        service = get_summarization_service()
        result = service.summarize(text)
    except Exception as e:
        print(f"获取模型服务过程中出错: {e}")
        sys.exit(1)
    return result['summary']
