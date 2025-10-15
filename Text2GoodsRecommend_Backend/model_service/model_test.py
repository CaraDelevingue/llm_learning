# test_summarization.py
import sys
import time
from model_loader import get_summarization_service

def run_tests():
    """运行测试用例"""
    
    try:
        # 获取服务实例
        print("🔄 初始化摘要服务...")
        service = get_summarization_service()
        print("✅ 服务初始化成功!")
        
        # 测试用例
        test_cases = [
            {
                "name": "科技新闻",
                "text": """
                人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。
                这些任务包括学习、推理、问题解决、感知和语言理解。近年来，深度学习技术的发展
                大大推动了人工智能领域的进步。深度学习使用多层神经网络来处理复杂的数据模式，
                在图像识别、自然语言处理和语音识别等领域取得了突破性进展。
                随着计算能力的提升和大数据的普及，人工智能技术正以前所未有的速度发展。
                """
            },
            {
                "name": "体育新闻", 
                "text": """
                在昨晚举行的NBA总决赛中，洛杉矶湖人队以108比105战胜了波士顿凯尔特人队，赢得了总冠军。
                勒布朗·詹姆斯表现出色，拿下了三双数据：32分、12个篮板和10次助攻。
                这场比赛充满了悬念，双方比分交替上升，直到最后时刻才分出胜负。
                这是湖人队历史上第18个总冠军，追平了凯尔特人队的纪录。
                """
            },
            {
                "name": "短文本测试",
                "text": "今天天气很好，阳光明媚，温度适宜，适合户外活动和运动。"
            },
            {
                "name": "学术概念-心理学",
                "text": """
                认知失调理论由心理学家费斯汀格提出，描述了个体在同时持有两种或多种心理认知时产生的不舒适感。
                这些认知可能包括信念、价值观、态度或行为信息。当这些认知之间存在矛盾时，个体就会体验到认知失调。
                为了减轻这种不适，人们会采取各种策略来恢复认知一致性。常见的策略包括改变其中一种认知、
                增加新的协调性认知、降低矛盾认知的重要性或改变自己的行为。例如，吸烟者明知吸烟有害健康却继续吸烟时，
                可能会通过告诉自己"吸烟能缓解压力"或"我认识的某人也吸烟但很长寿"等方式来减轻认知失调。
                这一理论在解释态度改变、决策过程和自我合理化等现象方面具有重要价值。
                """
            },
            {
                "name": "生活指南-烹饪方法",
                "text": """
                制作完美牛排的关键在于温度控制和肉质处理。首先选择厚度至少2.5厘米的牛排，从冰箱取出后在室温下放置30分钟。
                用厨房纸巾擦干表面水分，两面均匀涂抹橄榄油，撒上海盐和现磨黑胡椒。将厚底煎锅预热至高温，
                放入牛排后不要移动，第一面煎制2-3分钟直至形成焦脆外壳。翻面后继续煎制2-3分钟，此时可加入黄油、
                大蒜和迷迭香，倾斜锅体用勺子将融化的黄油反复淋在牛排表面。根据厚度和个人喜好调整时间：
                三分熟内部温度约52°C，五分熟约57°C，七分熟约63°C。煎好后将牛排放置在烤架上静置5-8分钟，
                让肉汁重新分布。切片时逆着纹理切割，可获得更嫩的口感。
                """
            },
            {
                "name": "长文本测试",
                "text": """
                中国空间站（天宫空间站）是一个模块化空间站系统，由中国独立建造和运营。
                它由天和核心舱、问天实验舱、梦天实验舱组成，总重量约100吨。
                空间站运行在约400公里的近地轨道，可支持3名航天员长期驻留。
                2022年完成在轨建造后，已进入应用与发展阶段，开展大量科学实验和技术试验。
                空间站的建设标志着中国载人航天工程迈入了新的历史阶段，为未来深空探测奠定了基础。
                中国空间站向国际合作伙伴开放，已有多个国家的科研项目入选。
                空间站的寿命设计为10年以上，期间将进行多次航天员轮换和物资补给。
                中国空间站的建设体现了中国航天技术的飞速发展和国家综合实力的提升。
                """
            }
        ]
        
        print("\n" + "="*60)
        print("开始测试摘要服务")
        print("="*60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 测试 {i}: {test_case['name']}")
            print("-" * 40)
            
            # 执行摘要
            start_time = time.time()
            result = service.summarize(
                test_case['text']
            )
            end_time = time.time()
            
            # 显示结果
            if result['success']:
                print(f"📄 原文长度: {result['original_length']} 字符")
                print(f"📊 摘要长度: {result['summary_length']} 字符")
                print(f"📉 压缩率: {result['compression_ratio']:.2%}")
                print(f"⏱️ 处理时间: {(end_time - start_time):.2f}秒")
                print(f"🔍 摘要内容: {result['summary']}")
            else:
                print(f"❌ 失败: {result['error']}")
            
            time.sleep(1)  # 避免连续请求
        
        # 批量测试
        print("\n" + "="*60)
        print("批量摘要测试")
        print("="*60)
        
        batch_texts = [
            "机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习并做出预测。",
            "气候变化是全球面临的重大挑战，需要各国共同努力减少温室气体排放。",
            "数字化转型正在改变传统行业的运营模式，提升效率和用户体验。"
        ]
        batch_results = service.batch_summarize(batch_texts)
        
        for j, (text, result) in enumerate(zip(batch_texts, batch_results), 1):
            print(f"\n批次 {j}:")
            print(f"输入: {text}")
            if result['success']:
                print(f"输出: {result['summary']}")
            else:
                print(f"错误: {result['error']}")
        
        # 性能测试
        print("\n" + "="*60)
        print("性能测试")
        print("="*60)
        
        performance_text = "自然语言处理是人工智能的重要分支，它使计算机能够理解、解释和生成人类语言。"
        
        times = []
        for _ in range(5):
            start = time.time()
            service.summarize(performance_text, max_length=50)
            end = time.time()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        print(f"平均处理时间: {avg_time:.3f}秒")
        print(f"最快: {min(times):.3f}秒")
        print(f"最慢: {max(times):.3f}秒")
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()