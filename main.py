import multiprocessing
from src.frontend.app import DashApp
from src.backend.websocket import app
import uvicorn

def run_dashboard():
    dash_app = DashApp()
    dash_app.app.run_server(host='0.0.0.0', port=8050, debug=True)

def run_websocket():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    dashboard_process = multiprocessing.Process(target=run_dashboard)
    websocket_process = multiprocessing.Process(target=run_websocket)
    
    dashboard_process.start()
    websocket_process.start()
    
    try:
        dashboard_process.join()
        websocket_process.join()
    except KeyboardInterrupt:
        dashboard_process.terminate()
        websocket_process.terminate()