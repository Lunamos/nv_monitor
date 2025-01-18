import torch
import time
import argparse
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiGPULoadTest:
    def __init__(self, gpu_ids: List[int], matrix_size: int = 5000):
        self.gpu_ids = gpu_ids
        self.matrix_size = matrix_size
        self.devices = [f'cuda:{i}' for i in gpu_ids]
        self.running = True
        
    def generate_load(self, device: str):
        """在指定GPU上生成计算负载"""
        try:
            # 创建大矩阵
            matrix1 = torch.randn(self.matrix_size, self.matrix_size, device=device)
            matrix2 = torch.randn(self.matrix_size, self.matrix_size, device=device)
            
            while self.running:
                # 矩阵乘法运算
                result = torch.matmul(matrix1, matrix2)
                # 确保计算完成
                torch.cuda.synchronize(device)
                
        except Exception as e:
            logger.error(f"GPU {device} 负载生成失败: {str(e)}")
            
    def start(self, duration: int = 60):
        """启动多GPU负载测试"""
        try:
            import threading
            threads = []
            
            # 为每个GPU创建负载线程
            for device in self.devices:
                thread = threading.Thread(
                    target=self.generate_load,
                    args=(device,)
                )
                thread.daemon = True
                threads.append(thread)
                
            # 启动所有线程
            logger.info(f"开始在 GPU {self.gpu_ids} 上生成负载")
            for thread in threads:
                thread.start()
                
            # 运行指定时长
            time.sleep(duration)
            
            # 停止所有线程
            self.running = False
            for thread in threads:
                thread.join()
                
            logger.info("负载测试完成")
            
        except Exception as e:
            logger.error(f"负载测试失败: {str(e)}")
        finally:
            self.running = False

def main():
    parser = argparse.ArgumentParser(description='多GPU负载测试')
    parser.add_argument('--gpus', type=int, nargs='+', required=True,
                      help='要测试的GPU ID列表，例如：0 1 2')
    parser.add_argument('--duration', type=int, default=60,
                      help='测试持续时间（秒）')
    parser.add_argument('--matrix-size', type=int, default=5000,
                      help='测试矩阵大小')
    
    args = parser.parse_args()
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        logger.error("CUDA不可用")
        return
        
    # 检查GPU ID是否有效
    available_gpus = list(range(torch.cuda.device_count()))
    for gpu_id in args.gpus:
        if gpu_id not in available_gpus:
            logger.error(f"GPU {gpu_id} 不存在")
            return
            
    # 启动测试
    test = MultiGPULoadTest(args.gpus, args.matrix_size)
    test.start(args.duration)

if __name__ == '__main__':
    main()