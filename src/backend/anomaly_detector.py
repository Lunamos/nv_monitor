from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class GPUStatus(Enum):
    NORMAL = "稳定"          # 正常
    FLUCTUATING = "波动"    # 波动
    CRITICAL = "故障"      # 严重

@dataclass
class StatusThreshold:
    # 温度阈值（摄氏度）
    temp_warning: float = 75.0      # 警告温度
    temp_critical: float = 85.0     # 严重温度
    
    # GPU利用率阈值
    util_low: float = 5.0          # 低负载
    util_normal: float = 80.0      # 正常负载
    util_high: float = 95.0        # 高负载
    
    # 利用率波动阈值（标准差）
    util_std_warning: float = 15.0  # 利用率波动警告
    
    # 功耗波动阈值（标准差）
    power_std_warning: float = 20.0 # 功耗波动警告
    
    # 显存使用率阈值
    memory_normal: float = 0.8     # 正常显存使用
    memory_high: float = 0.9       # 高显存使用
    memory_critical: float = 0.95   # 危险显存使用
    
    # ECC错误阈值
    ecc_warning: int = 1           # ECC错误警告阈值
    
    def detect(self, gpu_id: str) -> Tuple[GPUStatus, str]:
        """检测GPU状态，返回状态和原因"""
        if gpu_id not in self.history or not self.history[gpu_id]:
            return GPUStatus.CRITICAL, "无监控数据"
        
        history = self.history[gpu_id]
        latest = history[-1]
        
        # 检查严重问题
        critical_reasons = []
        
        # 检查nvidia-smi是否正常
        if 'nvidia_smi_ok' in latest and not latest['nvidia_smi_ok']:
            return GPUStatus.CRITICAL, "nvidia-smi 响应异常"
            
        # 检查供电异常
        if 'power_error' in latest and latest['power_error']:
            return GPUStatus.CRITICAL, f"供电异常: {latest['power_error']}"
            
        # 检查ECC错误
        if 'ecc_errors' in latest and latest['ecc_errors'] > 0:
            critical_reasons.append(f"ECC错误: {latest['ecc_errors']}次")
            
        # 检查温度
        if latest['temperature'] >= self.thresholds.temp_critical:
            critical_reasons.append(f"温度过高 ({latest['temperature']}°C)")
        
        # 检查显存
        memory_usage = latest['memory_used'] / latest['memory_total']
        if memory_usage >= self.thresholds.memory_critical:
            critical_reasons.append(f"显存使用率过高 ({memory_usage:.1%})")
            
        if critical_reasons:
            return GPUStatus.CRITICAL, " | ".join(critical_reasons)
        
        # 检查警告状态
        warnings = []
        
        # 检查利用率和功耗波动
        if len(history) >= 3:
            util_std = np.std([m['utilization_gpu'] for m in history])
            power_std = np.std([m['power_draw'] for m in history])
            
            if util_std >= self.thresholds.util_std_warning:
                warnings.append(f"利用率波动 (σ={util_std:.1f})")
            if power_std >= self.thresholds.power_std_warning:
                warnings.append(f"功耗波动 (σ={power_std:.1f})")
        
        # 检查温度警告
        if latest['temperature'] >= self.thresholds.temp_warning:
            warnings.append(f"温度偏高 ({latest['temperature']}°C)")
            
        # 检查显存警告
        if memory_usage >= self.thresholds.memory_high:
            warnings.append(f"显存使用率偏高 ({memory_usage:.1%})")
            
        if warnings:
            return GPUStatus.FLUCTUATING, " | ".join(warnings)
        
        # 正常状态描述
        status_desc = []
        util = latest['utilization_gpu']
        
        if util < self.thresholds.util_low:
            status_desc.append("低负载")
        elif util > self.thresholds.util_normal:
            status_desc.append("高负载")
        else:
            status_desc.append("正常负载")
            
        if memory_usage > self.thresholds.memory_normal:
            status_desc.append("高显存")
            
        if 'driver_version' in latest:
            status_desc.append(f"驱动 {latest['driver_version'].split()[0]}")
            
        return GPUStatus.NORMAL, " | ".join(status_desc) or "正常运行"

    def get_status_color(self, status: GPUStatus) -> str:
        """获取状态对应的莫兰迪色"""
        colors = {
            GPUStatus.NORMAL: "#C8D6CF",      # 柔和的绿色
            GPUStatus.FLUCTUATING: "#E6D0A7", # 柔和的黄色
            GPUStatus.CRITICAL: "#E6B8B8"     # 柔和的红色
        }
        return colors[status] 

class GPUAnomalyDetector:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self.thresholds = StatusThreshold()
    
    def update(self, gpu_id: str, metrics: Dict[str, Any]) -> None:
        """更新GPU历史数据"""
        if gpu_id not in self.history:
            self.history[gpu_id] = []
            
        self.history[gpu_id].append(metrics)
        if len(self.history[gpu_id]) > self.window_size:
            self.history[gpu_id].pop(0)
    
    def detect(self, gpu_id: str) -> Tuple[GPUStatus, str]:
        """检测GPU状态，返回状态和原因"""
        if gpu_id not in self.history or not self.history[gpu_id]:
            return GPUStatus.CRITICAL, "无监控数据"
        
        history = self.history[gpu_id]
        latest = history[-1]
        
        # 检查严重问题
        critical_reasons = []
        
        # 检查nvidia-smi是否正常
        if 'nvidia_smi_ok' in latest and not latest['nvidia_smi_ok']:
            return GPUStatus.CRITICAL, "nvidia-smi 响应异常"
            
        # 检查供电异常
        if 'power_error' in latest and latest['power_error']:
            return GPUStatus.CRITICAL, f"供电异常: {latest['power_error']}"
            
        # 检查ECC错误
        if 'ecc_errors' in latest and latest['ecc_errors'] > 0:
            critical_reasons.append(f"ECC错误: {latest['ecc_errors']}次")
            
        # 检查温度
        if latest['temperature'] >= self.thresholds.temp_critical:
            critical_reasons.append(f"温度过高 ({latest['temperature']}°C)")
        
        # 检查显存
        memory_usage = latest['memory_used'] / latest['memory_total']
        if memory_usage >= self.thresholds.memory_critical:
            critical_reasons.append(f"显存使用率过高 ({memory_usage:.1%})")
            
        if critical_reasons:
            return GPUStatus.CRITICAL, " | ".join(critical_reasons)
        
        # 检查警告状态
        warnings = []
        
        # 检查利用率和功耗波动
        if len(history) >= 3:
            util_std = np.std([m['utilization_gpu'] for m in history])
            power_std = np.std([m['power_draw'] for m in history])
            
            if util_std >= self.thresholds.util_std_warning:
                warnings.append(f"利用率波动 (σ={util_std:.1f})")
            if power_std >= self.thresholds.power_std_warning:
                warnings.append(f"功耗波动 (σ={power_std:.1f})")
        
        # 检查温度警告
        if latest['temperature'] >= self.thresholds.temp_warning:
            warnings.append(f"温度偏高 ({latest['temperature']}°C)")
            
        # 检查显存警告
        if memory_usage >= self.thresholds.memory_high:
            warnings.append(f"显存使用率偏高 ({memory_usage:.1%})")
            
        if warnings:
            return GPUStatus.FLUCTUATING, " | ".join(warnings)
        
        # 正常状态描述
        status_desc = []
        util = latest['utilization_gpu']
        
        if util < self.thresholds.util_low:
            status_desc.append("低负载")
        elif util > self.thresholds.util_normal:
            status_desc.append("高负载")
        else:
            status_desc.append("正常负载")
            
        if memory_usage > self.thresholds.memory_normal:
            status_desc.append("高显存")
            
        if 'driver_version' in latest:
            status_desc.append(f"驱动 {latest['driver_version'].split()[0]}")
            
        return GPUStatus.NORMAL, " | ".join(status_desc) or "正常运行"

    def get_status_color(self, status: GPUStatus) -> str:
        """获取状态对应的莫兰迪色"""
        colors = {
            GPUStatus.NORMAL: "#C8D6CF",      # 柔和的绿色
            GPUStatus.FLUCTUATING: "#E6D0A7", # 柔和的黄色
            GPUStatus.CRITICAL: "#E6B8B8"     # 柔和的红色
        }
        return colors[status]