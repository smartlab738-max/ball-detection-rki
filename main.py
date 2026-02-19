from flask import Flask
from core.system_manager import SystemManager
from web.routes import init_routes
import os

# Config Path
BASE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.join(BASE, 'config', 'nodes.json')

# Init System
print("--- SMART PINGPONG ADMIN ---")
manager = SystemManager(CONF)

# Init Web
app = Flask(__name__, template_folder='web/templates')
init_routes(app, manager)

if __name__ == '__main__':
    # Threaded=True WAJIB untuk video streaming
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)