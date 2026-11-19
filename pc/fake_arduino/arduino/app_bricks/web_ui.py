"""筆電上測試用：模擬 App Lab 的 WebUI brick（socket.io + 靜態網頁，port 7000）。"""
import socketio

sio = socketio.Server(async_mode="threading", cors_allowed_origins="*")


class WebUI:
    def __init__(self, *args, **kwargs):
        self.sio = sio

    def on_message(self, name, handler):
        sio.on(name)(lambda sid, data=None: handler(sid, data))

    def send_message(self, name, data):
        sio.emit(name, data)
