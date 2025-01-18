from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pynvml
import threading
import queue
import logging
from datetime import datetime
import time
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseMonitor(ABC):
    """监控基类，定义监控接口"""
    
    def __init__(self, buffer_size: int = 50):
        self.buffer_size = buffer_size
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.is_running = False
        self._monitor_thread = None

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标，需要子类实现"""
        pass

    def start(self):
        """启动监控线程"""
        if not self.is_running:
            self.is_running = True
            self._monitor_thread = threading.Thread(target=self._monitoring_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info(f"{self.__class__.__name__} 监控已启动")

    def stop(self):
        """停止监控线程"""
        self.is_running = False
        if self._monitor_thread:
            self._monitor_thread.join()
            logger.info(f"{self.__class__.__name__} 监控已停止")

    def _monitoring_loop(self):
        """监控主循环"""
        while self.is_running:
            try:
                metrics = self.get_metrics()
                if self.data_queue.full():
                    self.data_queue.get()
                self.data_queue.put((datetime.now(), metrics))
            except Exception as e:
                logger.error(f"监控出错: {str(e)}")
            time.sleep(1)

    def get_current_data(self) -> List[Dict[str, Any]]:
        """获取当前缓存的所有数据"""
        return list(self.data_queue.queue)

class GPUMonitor(BaseMonitor):
    """GPU监控实现"""
    
    def __init__(self, buffer_size: int = 50):
        super().__init__(buffer_size)
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"发现 {self.device_count} 个 GPU 设备")
        except Exception as e:
            logger.error(f"NVML初始化失败: {str(e)}")
            self.device_count = 0
        self.driver_version = self._get_driver_version()
        self._last_ecc_check = 0
        self._last_power_check = 0
        self._last_driver_check = 0
        
    def _run_command(self, command: str) -> Tuple[bool, str]:
        """运行shell命令并返回结果"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            return True, result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.error(f"命令执行超时: {command}")
            return False, ""
        except Exception as e:
            logger.error(f"命令执行失败: {str(e)}")
            return False, ""

    def _get_driver_version(self) -> Optional[str]:
        """获取显卡驱动版本"""
        success, output = self._run_command(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
        )
        if success and output:
            # 只取第一行，避免重复
            return output.split('\n')[0].strip()
        return None

    def _check_nvidia_smi_hang(self) -> bool:
        """检查nvidia-smi是否hang住"""
        success, _ = self._run_command("timeout -k 5s 5s nvidia-smi")
        return success

    def _check_power_errors(self) -> Optional[str]:
        """检查GPU供电异常"""
        success, output = self._run_command(
            "nvidia-smi -q -d POWER | grep 'Error'"
        )
        return output if success and output else None

    def _check_ecc_errors(self) -> Dict[str, int]:
        """检查ECC错误"""
        success, output = self._run_command(
            "nvidia-smi -q | grep -A2 'ECC Errors' | grep 'DRAM Uncorrectable'"
        )
        ecc_errors = {}
        if success and output:
            for i, count in enumerate(output.split('\n')):
                try:
                    count_str = count.split(':')[1].strip()
                    ecc_errors[f'gpu_{i}'] = int(count_str)
                except:
                    continue
        return ecc_errors

    def start(self):
        try:
            pynvml.nvmlInit()  # 确保在新进程中重新初始化
            super().start()
        except Exception as e:
            logger.error(f"NVML启动失败: {str(e)}")

    def stop(self):
        super().stop()
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    def get_metrics(self) -> Dict[str, Any]:
        """获取GPU监控指标"""
        metrics = {}
        current_time = time.time()
        
        try:
            # 基础指标获取（从原有代码）
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                metrics[f'gpu_{i}'] = self._get_basic_metrics(handle)
                
                # 添加额外的监控指标
                if current_time - self._last_ecc_check >= 60:  # 每分钟检查ECC
                    ecc_errors = self._check_ecc_errors()
                    if f'gpu_{i}' in ecc_errors:
                        metrics[f'gpu_{i}']['ecc_errors'] = ecc_errors[f'gpu_{i}']
                    self._last_ecc_check = current_time
                
                if current_time - self._last_power_check >= 300:  # 每5分钟检查供电
                    power_error = self._check_power_errors()
                    if power_error:
                        metrics[f'gpu_{i}']['power_error'] = power_error
                    self._last_power_check = current_time
                
                if current_time - self._last_driver_check >= 86400:  # 每天检查驱动
                    self.driver_version = self._get_driver_version()
                    self._last_driver_check = current_time
                
                metrics[f'gpu_{i}']['driver_version'] = self.driver_version
                metrics[f'gpu_{i}']['nvidia_smi_ok'] = self._check_nvidia_smi_hang()
                
        except Exception as e:
            logger.error(f"获取GPU指标失败: {str(e)}")
            
        return metrics

    def _get_basic_metrics(self, handle) -> Dict[str, Any]:
        """获取基础GPU指标"""
        return {
            'name': pynvml.nvmlDeviceGetName(handle),
            'memory_used': pynvml.nvmlDeviceGetMemoryInfo(handle).used // 1024 ** 2,
            'memory_total': pynvml.nvmlDeviceGetMemoryInfo(handle).total // 1024 ** 2,
            'memory_free': pynvml.nvmlDeviceGetMemoryInfo(handle).free // 1024 ** 2,
            'utilization_gpu': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            'utilization_memory': pynvml.nvmlDeviceGetUtilizationRates(handle).memory,
            'temperature': pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            'power_draw': pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        }