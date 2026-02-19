import json
import os
from ..camera_node import CameraNode
from ..ai_processor import AIProcessor

class SystemManager:
    def __init__(self, config_path):
        self.nodes = {}
        
        # Load AI Model Sekali Saja
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base, 'models', 'best.pt')
        self.ai = AIProcessor(model_path)

        self.load_config(config_path)

    def load_config(self, path):
        with open(path, 'r') as f:
            cfg = json.load(f)
        for nid, c in cfg.items():
            node = CameraNode(nid, c, self.ai)
            node.start_receiver()
            self.nodes[nid] = node

    def get_all_nodes(self): return self.nodes
    def get_node(self, nid): return self.nodes.get(nid)

    # Broadcast (Semua)
    def broadcast_command(self, action, w, h, fps):
        res = {}
        for nid, node in self.nodes.items():
            res[nid] = node.send_command(action, w, h, fps)
        return res

    # Single (Satu-satu)
    def single_command(self, nid, action, w, h, fps):
        node = self.nodes.get(nid)
        if node: return {nid: node.send_command(action, w, h, fps)}
        return {nid: "Not Found"}

    # AI Toggle
    def set_ai_mode(self, target, enabled):
        if target == 'ALL':
            for node in self.nodes.values(): node.set_ai_status(enabled)
        else:
            node = self.nodes.get(target)
            if node: node.set_ai_status(enabled)
        return {"status": "ok", "ai": enabled}