from fastapi import FastAPI, WebSocket
from typing import List, Dict, Any
import asyncio
import json
import logging
from .monitor import GPUMonitor
import pynvml
from .anomaly_detector import GPUAnomalyDetector

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.gpu_monitor = GPUMonitor()
        self.anomaly_detector = GPUAnomalyDetector()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, data: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                metrics = data['metrics']
                status_info = {}
                for gpu_id, gpu_metrics in metrics.items():
                    self.anomaly_detector.update(gpu_id, gpu_metrics)
                    status, reason = self.anomaly_detector.detect(gpu_id)
                    status_info[gpu_id] = {
                        'status': status.value,
                        'reason': reason
                    }
                
                data['status'] = status_info
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"发送数据失败: {str(e)}")
                
    async def start_monitoring(self):
        self.gpu_monitor.start()
        while True:
            try:
                data = self.gpu_monitor.get_current_data()
                if data:
                    latest_data = data[-1]
                    await self.broadcast({
                        'timestamp': latest_data[0].isoformat(),
                        'metrics': latest_data[1]
                    })
            except Exception as e:
                logger.error(f"监控数据广播失败: {str(e)}")
            await asyncio.sleep(1)

manager = WebSocketManager()
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    try:
        pynvml.nvmlInit()
        asyncio.create_task(manager.start_monitoring())
    except Exception as e:
        logger.error(f"NVML启动失败: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    try:
        pynvml.nvmlShutdown()
    except:
        pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)