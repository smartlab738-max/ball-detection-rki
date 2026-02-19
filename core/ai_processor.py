from ultralytics import YOLO
import os
import cv2
import json
import numpy as np
from .court_detector import CourtDetector
from .zone_mapper import ZoneMapper

class AIProcessor:
    def __init__(self, model_path, camera_id):
        self.camera_id = camera_id
        
        # Inisialisasi Model YOLO
        if not os.path.exists(model_path):
            print(f"[AI-{camera_id}] ERROR: Model tidak ditemukan.")
            self.model = None
        else:
            self.model = YOLO(model_path)

        # Inisialisasi Modul Deteksi Lapangan & Zona
        self.court = CourtDetector()
        self.zone_mapper = None
        self.show_court_overlay = True
        self.show_zone_grid = True
        
        # Muat kalibrasi jika sudah ada di folder config
        self._load_calibration()

    def _load_calibration(self):
        cal_path = f"config/calibration/{self.camera_id}_homography.json"
        if os.path.exists(cal_path):
            try:
                with open(cal_path, 'r') as f:
                    cal_data = json.load(f)
                
                self.court.corners = np.array(cal_data['corners'], dtype=np.float32)
                self.court.matrix = np.array(cal_data['matrix'], dtype=np.float32)
                self.court.is_calibrated = True
                
                table_width = cal_data.get('table_width', 274.0)
                table_height = cal_data.get('table_height', 152.5)
                self.zone_mapper = ZoneMapper(self.court.matrix, table_width, table_height)
                print(f"[AI-{self.camera_id}] ✅ Kalibrasi dimuat.")
            except Exception as e:
                print(f"[AI-{self.camera_id}] ❌ Gagal memuat kalibrasi: {e}")

    def auto_calibrate(self, frame):
        """Memicu deteksi otomatis kontur meja dari frame saat ini."""
        print(f"[AI-{self.camera_id}] Mencoba deteksi otomatis...")
        corners = self.court.auto_detect_corners(frame)
        
        if corners is not None:
            self.court.compute_homography()
            self.zone_mapper = ZoneMapper(self.court.matrix)
            
            # Simpan hasil kalibrasi ke file JSON
            os.makedirs("config/calibration", exist_ok=True)
            self.court.save_calibration(f"config/calibration/{self.camera_id}_homography.json")
            return True
        return False

    def process(self, frame):
        if self.model is None:
            return frame

        try:
            # 1. Deteksi YOLO
            results = self.model(frame, conf=0.35, iou=0.5, verbose=False)
            annotated = results[0].plot()
            
            # 2. Gambar Overlay Lapangan
            if self.court.is_calibrated:
                if self.show_court_overlay:
                    annotated = self.court.overlay_court_lines(annotated)
                if self.show_zone_grid:
                    annotated = self.court.overlay_zone_grid(annotated)
                
                # 3. Mapping Bola ke Zona
                if self.zone_mapper is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        if cls_id == 0:  # Anggap 0 adalah bola pingpong
                            cx, cy = float(box.xywh[0][0]), float(box.xywh[0][1])
                            zone_id = self.zone_mapper.get_zone_id(cx, cy)
                            
                            if zone_id >= 0:
                                self.zone_mapper.update_stats(zone_id)
                                annotated = self.zone_mapper.draw_zone_label(annotated, cx, cy, zone_id)
            return annotated
        except Exception as e:
            print(f"Error: {e}")
            return frame

    def toggle_court_overlay(self, enabled): self.show_court_overlay = enabled
    def toggle_zone_grid(self, enabled): self.show_zone_grid = enabled
    def get_zone_stats(self): return self.zone_mapper.export_stats() if self.zone_mapper else None
    def reset_zone_stats(self):
        if self.zone_mapper: self.zone_mapper.reset_stats()