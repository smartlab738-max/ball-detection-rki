import json
import os
from .camera_node import CameraNode
from .ai_processor import AIProcessor

class SystemManager:
    def __init__(self, config_path):
        self.nodes = {}
        # Mendapatkan path absolut ke folder models
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.load_config(config_path)

    def load_config(self, path):
        """Memuat konfigurasi node dan inisialisasi AI per kamera."""
        with open(path, 'r') as f:
            cfg = json.load(f)
        
        # Path model YOLOv8
        # model_path = os.path.join(self.base_path, 'models', 'best.pt')
        
        # Path model openvino
        model_path = os.path.join(self.base_path, 'models', 'best_openvino_model')

        for nid, c in cfg.items():
            # Inisialisasi AIProcessor unik untuk setiap ID kamera
            local_ai = AIProcessor(model_path, camera_id=nid)
            
            # Buat instance CameraNode dengan AI lokalnya
            node = CameraNode(nid, c, local_ai)
            node.start_receiver()
            self.nodes[nid] = node

    def get_all_nodes(self): 
        return self.nodes

    def get_node(self, nid): 
        return self.nodes.get(nid)

    # --- FUNGSI KONTROL YANG ERROR TADI ---

    def broadcast_command(self, action, w, h, fps):
        """Mengirim perintah ke SEMUA node kamera."""
        res = {}
        for nid, node in self.nodes.items():
            res[nid] = node.send_command(action, w, h, fps)
        return res

    def single_command(self, nid, action, w, h, fps):
        """Mengirim perintah ke SATU node kamera tertentu."""
        node = self.nodes.get(nid)
        if node:
            return {nid: node.send_command(action, w, h, fps)}
        return {nid: "Not Found"}

    def set_ai_mode(self, target, enabled):
        """Mengaktifkan atau mematikan mode deteksi AI."""
        if target == 'ALL':
            for node in self.nodes.values(): 
                node.set_ai_status(enabled)
        else:
            node = self.nodes.get(target)
            if node: 
                node.set_ai_status(enabled)
        return {"status": "ok", "ai": enabled}