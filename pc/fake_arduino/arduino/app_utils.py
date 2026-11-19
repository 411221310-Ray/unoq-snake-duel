"""筆電上測試用：模擬 App Lab 的 App / Bridge。LED 畫面存在 last_frame，App.run() 開一個 port 7000 的網頁伺服器。"""
import os
import sys

last_frame = None


class Bridge:
    @staticmethod
    def call(name, data, timeout=2):
        global last_frame
        if name == "draw":
            last_frame = bytes(data)


class App:
    @staticmethod
    def run():
        import socketio
        from werkzeug.serving import run_simple

        from arduino.app_bricks.web_ui import sio

        assets = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "..", "assets")
        app = socketio.WSGIApp(sio, static_files={"/": {"content_type": "text/html", "filename": os.path.join(assets, "index.html")},
                                                  "/libs": os.path.join(assets, "libs"),
                                                  "/app.js": os.path.join(assets, "app.js"),
                                                  "/style.css": os.path.join(assets, "style.css")})
        port = int(os.environ.get("PORT", "7000"))
        print(f"假板子網頁：http://localhost:{port}", flush=True)
        run_simple("0.0.0.0", port, app, threaded=True, use_reloader=False)
