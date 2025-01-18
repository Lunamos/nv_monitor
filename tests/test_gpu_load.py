import torch
import time
import random

def create_random_load():
    # 创建大矩阵占用显存
    tensors = []
    while True:
        try:
            # 随机大小的张量，最大 2GB
            size = random.randint(1000, 10000)
            tensor = torch.randn(size, size, device='cuda')
            tensors.append(tensor)
            
            # 随机矩阵乘法造成计算负载
            if random.random() < 0.3:
                result = torch.matmul(tensor, tensor)
                
            # 随机暂停造成波动
            time.sleep(random.uniform(0.1, 0.5))
            
        except RuntimeError:  # 显存不足时释放部分内存
            if tensors:
                tensors.pop(0)
            time.sleep(1)

if __name__ == "__main__":
    create_random_load() 