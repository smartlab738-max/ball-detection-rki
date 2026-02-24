# from ultralytics import YOLO
# import os
# import cv2
# import json
# import numpy as np
# from collections import deque
# from .court_detector import CourtDetector
# from .zone_mapper import ZoneMapper

# class AIProcessor:
#     def __init__(self, model_path, camera_id):
#         self.camera_id = camera_id
        
#         # Inisialisasi Model YOLO OpenVINO
#         if not os.path.exists(model_path):
#             print(f"[AI-{camera_id}] ERROR: Model tidak ditemukan di path: {model_path}")
#             self.model = None
#         else:
#             self.model = YOLO(model_path)

#         # Inisialisasi Modul Deteksi Lapangan & Zona
#         self.court = CourtDetector()
#         self.zone_mapper = None
#         self.show_court_overlay = True
#         self.show_zone_grid = True
        
#         # === TAHAP 2: VARIABEL FISIKA & PANTULAN ===
#         # Menyimpan 8 posisi bola terakhir (X, Y) untuk membuat "ekor komet"
#         self.ball_history = deque(maxlen=8) 
        
#         # Timer untuk memunculkan tulisan "BOUNCE!" di layar selama beberapa frame
#         self.bounce_timer = 0 
#         self.last_bounce_pos = (0, 0)
        
#         # Muat kalibrasi jika sudah ada di folder config
#         self._load_calibration()

#     def _load_calibration(self):
#         cal_path = f"config/calibration/{self.camera_id}_homography.json"
#         if os.path.exists(cal_path):
#             try:
#                 with open(cal_path, 'r') as f:
#                     cal_data = json.load(f)
                
#                 self.court.corners = np.array(cal_data['corners'], dtype=np.float32)
#                 self.court.matrix = np.array(cal_data['matrix'], dtype=np.float32)
#                 self.court.is_calibrated = True
                
#                 self.court._update_drawing_cache()
                
#                 table_width = cal_data.get('table_width', 274.0)
#                 table_height = cal_data.get('table_height', 152.5)
#                 self.zone_mapper = ZoneMapper(self.court.matrix, table_width, table_height)
#                 print(f"[AI-{self.camera_id}] ✅ Kalibrasi dimuat.")
#             except Exception as e:
#                 print(f"[AI-{self.camera_id}] ❌ Gagal memuat kalibrasi: {e}")

#     def auto_calibrate(self, frame):
#         """Memicu deteksi otomatis kontur meja dari frame saat ini."""
#         print(f"[AI-{self.camera_id}] Mencoba deteksi otomatis...")
#         corners = self.court.auto_detect_corners(frame)
        
#         if corners is not None:
#             self.court.compute_homography()
#             self.zone_mapper = ZoneMapper(self.court.matrix)
#             os.makedirs("config/calibration", exist_ok=True)
#             self.court.save_calibration(f"config/calibration/{self.camera_id}_homography.json")
#             return True
#         return False

#     def process(self, frame):
#         if self.model is None: return frame

#         try:
#             # 1. Deteksi YOLO (Gunakan conf 0.15 agar bola ngeblur tetap kena)
#             results = self.model(frame, conf=0.15, iou=0.4, imgsz=320, verbose=False)
#             annotated = results[0].plot()
            
#             ball_detected_this_frame = False

#             # 2. Gambar Overlay Lapangan
#             if self.court.is_calibrated:
#                 if self.show_court_overlay:
#                     annotated = self.court.overlay_court_lines(annotated)
#                 if self.show_zone_grid:
#                     annotated = self.court.overlay_zone_grid(annotated)
                
#                 # Cek jika ada bola
#                 for box in results[0].boxes:
#                     cls_id = int(box.cls[0])
#                     if cls_id == 0:  
#                         ball_detected_this_frame = True
#                         cx, cy = float(box.xywh[0][0]), float(box.xywh[0][1])
                        
#                         # Simpan jejak bola ke dalam memori
#                         self.ball_history.append((int(cx), int(cy)))
                        
#                         # Hitung Statistik Zona
#                         if self.zone_mapper is not None:
#                             zone_id = self.zone_mapper.get_zone_id(cx, cy)
#                             if zone_id >= 0:
#                                 self.zone_mapper.update_stats(zone_id)
#                                 annotated = self.zone_mapper.draw_zone_label(annotated, cx, cy, zone_id)

#             # 3. ALGORITMA PANTULAN (V-SHAPE BOUNCE)
#             if len(self.ball_history) >= 3 and ball_detected_this_frame:
#                 # Ambil 3 titik kordinat Y (Ketinggian) yang paling baru
#                 y1 = self.ball_history[-3][1] # Titik lama
#                 y2 = self.ball_history[-2][1] # Titik tengah (potensi pantulan)
#                 y3 = self.ball_history[-1][1] # Titik paling baru

#                 # Hitung selisih
#                 dy_sebelum = y2 - y1
#                 dy_sesudah = y3 - y2

#                 # Syarat Pantulan: Sebelumnya turun (Positif), sekarang naik (Negatif)
#                 # Angka 2 adalah batas toleransi piksel (mencegah getaran kamera dihitung pantulan)
#                 if dy_sebelum > 2 and dy_sesudah < -2:
#                     self.bounce_timer = 15 # Munculkan efek ledakan selama 15 frame
#                     self.last_bounce_pos = self.ball_history[-2] # Catat lokasi tepat saat memantul
                    
#                     # (Opsional) Disini nanti kita bisa memicu Auto-Save VAR secara otomatis!

#             # 4. EFEK VISUAL TAHAP 2
#             # A. Gambar Ekor Komet (Jejak pergerakan bola)
#             for i in range(1, len(self.ball_history)):
#                 pt1 = self.ball_history[i-1]
#                 pt2 = self.ball_history[i]
#                 # Gambar garis kuning menyambung titik-titik sebelumnya
#                 cv2.line(annotated, pt1, pt2, (0, 255, 255), 2)

#             # B. Gambar Ledakan "BOUNCE!" jika sedang memantul
#             if self.bounce_timer > 0:
#                 bx, by = self.last_bounce_pos
#                 # Gambar Titik Merah
#                 cv2.circle(annotated, (bx, by), 15, (0, 0, 255), -1) 
#                 # Tulis BOUNCE!
#                 cv2.putText(annotated, "BOUNCE!", (bx - 50, by - 25), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
#                 self.bounce_timer -= 1 # Kurangi timer

#             return annotated
            
#         except Exception as e:
#             print(f"[AI-{self.camera_id}] Error Process AI: {e}")
#             return frame

#     def toggle_court_overlay(self, enabled): self.show_court_overlay = enabled
#     def toggle_zone_grid(self, enabled): self.show_zone_grid = enabled
#     def get_zone_stats(self): return self.zone_mapper.export_stats() if self.zone_mapper else None
#     def reset_zone_stats(self):
#         if self.zone_mapper: self.zone_mapper.reset_stats()



from ultralytics import YOLO
import os
import cv2
import json
import numpy as np
from collections import deque
from .court_detector import CourtDetector
from .zone_mapper import ZoneMapper

class AIProcessor:
    def __init__(self, model_path, camera_id):
        self.camera_id = camera_id
        
        # Inisialisasi Model YOLO
        if not os.path.exists(model_path):
            print(f"[AI-{camera_id}] ERROR: Model tidak ditemukan di path: {model_path}")
            self.model = None
        else:
            self.model = YOLO(model_path)

        # Inisialisasi Modul Deteksi Lapangan & Zona
        self.court = CourtDetector()
        self.zone_mapper = None
        self.show_court_overlay = True
        self.show_zone_grid = True
        
        # === TAHAP 2: VARIABEL FISIKA & PANTULAN ===
        # Menyimpan 8 posisi bola terakhir untuk "ekor komet"
        self.ball_history = deque(maxlen=8) 
        
        # Timer untuk efek animasi pantulan
        self.bounce_timer = 0 
        self.last_bounce_pos = (0, 0)
        
        # Muat kalibrasi jika sudah ada
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
                
                self.court._update_drawing_cache()
                
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
            os.makedirs("config/calibration", exist_ok=True)
            self.court.save_calibration(f"config/calibration/{self.camera_id}_homography.json")
            return True
        return False

    def process(self, frame):
        if self.model is None: return frame

        try:
            # 1. Deteksi YOLO (Conf diturunkan agar bola samar tetap terdeteksi)
            results = self.model(frame, conf=0.4, iou=0.4, imgsz=320, verbose=False)
            
            # Kita tidak lagi menggunakan plot() bawaan ultralytics agar kotak deteksinya tidak tebal
            annotated = frame.copy()
            
            ball_detected_this_frame = False

            # 2. Gambar Overlay Lapangan
            if self.court.is_calibrated:
                if self.show_court_overlay:
                    annotated = self.court.overlay_court_lines(annotated)
                if self.show_zone_grid:
                    annotated = self.court.overlay_zone_grid(annotated)
                
                # Cek hasil deteksi bola
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  
                        ball_detected_this_frame = True
                        
                        # Ambil koordinat titik tengah dan ukuran bola
                        cx, cy = float(box.xywh[0][0]), float(box.xywh[0][1])
                        w, h = float(box.xywh[0][2]), float(box.xywh[0][3])
                        
                        # OPTIMASI AKURASI: Gunakan titik BAWAH bola sebagai acuan sentuhan meja, bukan titik tengahnya
                        bottom_y = cy + (h / 2)
                        
                        # Simpan jejak bola ke dalam memori
                        self.ball_history.append((int(cx), int(bottom_y)))
                        
                        # Gambar bola dengan lingkaran kecil yang rapi (Bukan kotak tebal)
                        cv2.circle(annotated, (int(cx), int(cy)), int(w/2), (0, 255, 0), 2)
                        
                        # Hitung Statistik Zona
                        if self.zone_mapper is not None:
                            zone_id = self.zone_mapper.get_zone_id(cx, bottom_y)
                            if zone_id >= 0:
                                self.zone_mapper.update_stats(zone_id)
                                annotated = self.zone_mapper.draw_zone_label(annotated, cx, cy, zone_id)

            # 3. ALGORITMA PANTULAN (V-SHAPE BOUNCE)
            if len(self.ball_history) >= 3 and ball_detected_this_frame:
                y1 = self.ball_history[-3][1] 
                y2 = self.ball_history[-2][1] # Titik pantulan terendah
                y3 = self.ball_history[-1][1] 

                dy_sebelum = y2 - y1
                dy_sesudah = y3 - y2

                # Syarat Pantulan (Toleransi 2 piksel)
                if dy_sebelum > 2 and dy_sesudah < -2:
                    self.bounce_timer = 15 # Durasi animasi ripple (15 frame)
                    self.last_bounce_pos = self.ball_history[-2] 

            # 4. EFEK VISUAL TAHAP 2 (DIPERHALUS)
            
            # A. Gambar Ekor Komet Halus
            for i in range(1, len(self.ball_history)):
                pt1 = self.ball_history[i-1]
                pt2 = self.ball_history[i]
                
                # Ketebalan menipis di ujung ekor (dari 1px membesar ke 3px dekat bola)
                thickness = int((i / len(self.ball_history)) * 3) + 1
                
                # Warna biru muda/cyan agar lebih estetik dan tidak mencolok
                cv2.line(annotated, pt1, pt2, (255, 255, 0), thickness, cv2.LINE_AA)

            # B. Animasi Ripple Pantulan (Tanpa Teks Raksasa)
            if self.bounce_timer > 0:
                bx, by = self.last_bounce_pos
                
                # Hitung radius lingkaran yang membesar seiring berkurangnya timer
                # Timer mundur dari 15 ke 0. Radius membesar dari 0 ke 30.
                radius = (15 - self.bounce_timer) * 2 
                
                # Gambar lingkaran oranye yang membesar
                cv2.circle(annotated, (bx, by), radius, (0, 165, 255), 2, cv2.LINE_AA) 
                # Gambar titik merah kecil tepat di pusat pantulan
                cv2.circle(annotated, (bx, by), 3, (0, 0, 255), -1) 
                
                self.bounce_timer -= 1 

            return annotated
            
        except Exception as e:
            print(f"[AI-{self.camera_id}] Error Process AI: {e}")
            return frame

    def toggle_court_overlay(self, enabled): self.show_court_overlay = enabled
    def toggle_zone_grid(self, enabled): self.show_zone_grid = enabled
    def get_zone_stats(self): return self.zone_mapper.export_stats() if self.zone_mapper else None
    def reset_zone_stats(self):
        if self.zone_mapper: self.zone_mapper.reset_stats()