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
        
        # Path model openvino
        model_path = os.path.join(self.base_path, 'models', 'best_openvino_model')
        
        # Path model YOLOv8
        # model_path = os.path.join(self.base_path, 'models', 'best.pt')

        for nid, c in cfg.items():
            # Inisialisasi AIProcessor unik untuk setiap ID kamera
            local_ai = AIProcessor(model_path, camera_id=nid)
            
            # Buat instance CameraNode dengan AI lokalnya
            node = CameraNode(nid, c, local_ai)
            
            # Mulai menangkap video
            node.start_receiver()
            self.nodes[nid] = node

    def get_all_nodes(self): 
        return self.nodes

    def get_node(self, nid): 
        return self.nodes.get(nid)

    def stop_all(self):
        """Fungsi utilitas untuk mematikan semua thread saat aplikasi Flask ditutup."""
        for node in self.nodes.values():
            node.stop_receiver()

    # --- FUNGSI KONTROL YANG SUDAH DI-OPTIMASI ---

    def broadcast_command(self, action, w, h, fps):
        """Mengirim perintah ke SEMUA node kamera dan mengatur thread lokal."""
        res = {}
        act_lower = action.lower()
        
        for nid, node in self.nodes.items():
            # 1. Kirim perintah ke Raspberry Pi via TCP
            res[nid] = node.send_command(action, w, h, fps)
            
            # 2. Manajemen Thread Lokal (Optimasi 3)
            if act_lower in ['stop', 'restart_service', 'reboot_pi', 'shutdown_pi']:
                node.stop_receiver() # Matikan proses pembacaan video di laptop
            elif act_lower == 'start':
                if not node.running:
                    node.start_receiver() # Hidupkan kembali jika sebelumnya mati
                    
        return res

    def single_command(self, nid, action, w, h, fps):
        """Mengirim perintah ke SATU node kamera tertentu dan mengatur thread lokal."""
        node = self.nodes.get(nid)
        act_lower = action.lower()
        
        if node:
            # 1. Kirim perintah ke Raspberry Pi via TCP
            response = node.send_command(action, w, h, fps)
            
            # 2. Manajemen Thread Lokal (Optimasi 3)
            if act_lower in ['stop', 'restart_service', 'reboot_pi', 'shutdown_pi']:
                node.stop_receiver() # Matikan proses pembacaan video di laptop
            elif act_lower == 'start':
                if not node.running:
                    node.start_receiver() # Hidupkan kembali jika sebelumnya mati
                    
            return {nid: response}
            
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