from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from datetime import datetime
import json
import websockets
import asyncio
import threading
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DashApp:
    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.data_buffer = {}
        self.timestamps = []
        self.max_points = 50
        
        self.setup_layout()
        self.setup_callbacks()
        self.start_ws_client()
        
    def setup_layout(self):
        self.app.layout = html.Div([
            # 导航栏
            dbc.Navbar(
                dbc.Container([
                    html.A(
                        dbc.Row([
                            dbc.Col(html.H3("InfPlane", className="ms-2 text-white")),
                        ]),
                        href="/",
                        style={"textDecoration": "none"},
                    )
                ]),
                color="dark",
                dark=True,
            ),
            
            # 主要内容区域
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("GPU监控", className="text-center")),
                            dbc.CardBody([
                                html.Div(id="gpu-info"),
                                dbc.Tabs([
                                    dbc.Tab(dcc.Graph(id="gpu-utilization"), label="GPU利用率"),
                                    dbc.Tab(dcc.Graph(id="gpu-memory"), label="显存使用"),
                                    dbc.Tab(dcc.Graph(id="gpu-temperature"), label="温度"),
                                    dbc.Tab(dcc.Graph(id="gpu-power"), label="功耗"),
                                ])
                            ])
                        ], className="mb-4 shadow-sm")
                    ], width=12)
                ])
            ], className="mt-4"),
            
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
        ])
        
    def setup_callbacks(self):
        @self.app.callback(
            [Output("gpu-info", "children"),
             Output("gpu-utilization", "figure"),
             Output("gpu-memory", "figure"),
             Output("gpu-temperature", "figure"),
             Output("gpu-power", "figure")],
            Input('interval-component', 'n_intervals')
        )
        def update_graphs(n):
            if not self.data_buffer:
                return html.Div("等待GPU数据..."), {}, {}, {}, {}

            # GPU基本信息
            gpu_info = []
            for gpu_id in sorted(self.data_buffer.keys()):
                if self.data_buffer[gpu_id]:
                    latest = self.data_buffer[gpu_id][-1]
                    status = latest.get('status', {
                        'status': 'normal',  # 默认值为 normal
                        'color': '#C8D6CF'
                    })
                    
                    # 根据状态设置Alert的颜色
                    status_color = {
                        '稳定': 'success',
                        '波动': 'warning',
                        '故障': 'danger'
                    }.get(status['status'], 'secondary')
                    
                    gpu_info.append(
                        dbc.Alert([
                            html.H5([
                                f"GPU {gpu_id}: {latest['name']}",
                                html.Span(
                                    f"{status['status']} ({status.get('reason', '')})",
                                    className="ms-2"
                                )
                            ]),
                            html.P([
                                f"温度: {latest['temperature']}°C | ",
                                f"显存: {latest['memory_used']}/{latest['memory_total']} MiB | ",
                                f"功耗: {latest['power_draw']:.1f}W | ",
                                f"利用率: {latest['utilization_gpu']}%"
                            ], className="mb-0"),
                            # 添加新的监控信息显示
                            html.P([
                                html.Small([
                                    f"驱动版本: {latest.get('driver_version', '未知')} | ",
                                    "供电状态: " + ("正常" if not latest.get('power_error') else f"异常: {latest['power_error']}") + " | ",
                                    f"ECC错误: {latest.get('ecc_errors', 0)}次 | ",
                                    "nvidia-smi: " + ("正常" if latest.get('nvidia_smi_ok', True) else "异常")
                                ], className="text-muted")
                            ], className="mb-0 mt-1")
                        ], color=status_color, className="mb-3")
                    )

            # 创建图表
            figures = []
            metrics = [
                ('utilization_gpu', 'GPU利用率', '%'),
                ('memory_used', '显存使用', 'MiB'),
                ('temperature', '温度', '°C'),
                ('power_draw', '功耗', 'W')
            ]

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

            for metric, title, unit in metrics:
                fig = go.Figure()
                for i, gpu_id in enumerate(sorted(self.data_buffer.keys())):
                    if not self.data_buffer[gpu_id]:
                        continue
                    
                    fig.add_trace(go.Scatter(
                        x=self.timestamps[-self.max_points:],
                        y=[d[metric] for d in self.data_buffer[gpu_id][-self.max_points:]],
                        name=f'GPU {gpu_id}',
                        line=dict(color=colors[i % len(colors)])
                    ))
                    
                fig.update_layout(
                    title=title,
                    xaxis_title='时间',
                    yaxis_title=unit,
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                figures.append(fig)

            return html.Div(gpu_info), *figures
            
    async def websocket_client(self):
        uri = "ws://localhost:8000/ws"
        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    logger.info("WebSocket 连接成功")
                    await websocket.send("start")
                    while True:
                        data = await websocket.recv()
                        data = json.loads(data)
                        timestamp = datetime.fromisoformat(data['timestamp'])
                        self.timestamps.append(timestamp)

                        # 动态更新GPU数据缓冲区
                        for gpu_id, metrics in data['metrics'].items():
                            if gpu_id not in self.data_buffer:
                                self.data_buffer[gpu_id] = []
                            # 添加状态信息到 metrics
                            if 'status' in data:
                                metrics['status'] = data['status'][gpu_id]
                            self.data_buffer[gpu_id].append(metrics)
                            if len(self.data_buffer[gpu_id]) > self.max_points:
                                self.data_buffer[gpu_id].pop(0)

                        if len(self.timestamps) > self.max_points:
                            self.timestamps.pop(0)
                            
            except Exception as e:
                logger.error(f"WebSocket连接失败: {str(e)}")
                await asyncio.sleep(1)
                
    def start_ws_client(self):
        def run_async_client():
            asyncio.run(self.websocket_client())
            
        thread = threading.Thread(target=run_async_client, daemon=True)
        thread.start()
        
    def run(self, debug=False, port=8050):
        self.app.run_server(debug=debug, port=port)