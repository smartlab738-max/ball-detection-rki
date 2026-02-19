import cv2
import numpy as np
import argparse
import os
import sys
import requests

# Menambahkan path agar bisa import dari folder core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.court_detector import CourtDetector

class CalibrationTool:
    def __init__(self, camera_id, video_source):
        self.camera_id = camera_id
        # Jika video_source hanya angka, ubah ke int (webcam lokal)
        # Jika string, gunakan sebagai URL/Path
        try:
            self.video_source = int(video_source)
        except:
            self.video_source = video_source
        
        self.detector = CourtDetector()
        self.points = []
        self.frame = None
        self.display_frame = None
        
        self.window_name = f"Calibration Tool - {self.camera_id}"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Menangkap klik mouse untuk menentukan 4 sudut meja."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
                print(f"[CAL] Titik {len(self.points)}: ({x}, {y})")
                
                if len(self.points) == 4:
                    print("[CAL] ✅ 4 Sudut terpilih. Menghitung Homografi...")
                    self.apply_calibration()

    def apply_calibration(self):
        """Menerapkan kalibrasi manual ke detektor."""
        try:
            self.detector.set_manual_corners(self.points)
            self.detector.compute_homography(scale_factor=1.0) # Menggunakan dimensi asli cm
            print("[CAL] ✅ Kalibrasi berhasil diaplikasikan.")
        except Exception as e:
            print(f"[CAL] ❌ Error: {e}")

    def draw_ui(self):
        """Menggambar instruksi dan titik tanpa menutupi area utama meja."""
        if self.frame is None: return
        self.display_frame = self.frame.copy()
        h, w = self.display_frame.shape[:2]
        
        # 1. Buat Sidebar Gelap di sisi kiri agar teks tidak menumpuk di gambar
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (220, h), (0, 0, 0), -1)
        # Gabungkan dengan transparansi agar meja di bawahnya masih samar terlihat
        cv2.addWeighted(overlay, 0.6, self.display_frame, 0.4, 0, self.display_frame)
        
        # 2. Gambar instruksi di area sidebar
        instructions = [
            f"Kamera: {self.camera_id}",
            "--- URUTAN KLIK ---",
            "1. Kiri Atas",
            "2. Kanan Atas",
            "3. Kanan Bawah",
            "4. Kiri Bawah",
            "-------------------",
            "Titik Terpilih: " + str(len(self.points)),
            "-------------------",
            "[s] Simpan",
            "[r] Reset",
            "[q] Keluar"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(self.display_frame, text, (10, 30 + i*25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Gambar titik dan garis
        for i, pt in enumerate(self.points):
            # Lingkaran titik
            cv2.circle(self.display_frame, tuple(pt), 6, (0, 255, 0), -1)
            cv2.circle(self.display_frame, tuple(pt), 8, (255, 255, 255), 1)
            # Label angka di dekat titik
            cv2.putText(self.display_frame, str(i+1), (pt[0]+10, pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Gambar garis penghubung jika sudah ada minimal 2 titik
        if len(self.points) >= 2:
            for i in range(len(self.points)):
                if i + 1 < len(self.points):
                    cv2.line(self.display_frame, tuple(self.points[i]), tuple(self.points[i+1]), (0, 255, 255), 2)
                elif len(self.points) == 4: # Tutup garis kalau sudah 4 titik
                    cv2.line(self.display_frame, tuple(self.points[3]), tuple(self.points[0]), (0, 255, 255), 2)

        # 4. Tampilkan overlay lapangan jika kalibrasi aktif
        if self.detector.is_calibrated:
            self.display_frame = self.detector.overlay_court_lines(self.display_frame)

    def save_calibration(self):
        """Menyimpan hasil kalibrasi ke file JSON."""
        if not self.detector.is_calibrated:
            print("[CAL] ❌ Gagal Simpan: Belum dikalibrasi!")
            return
        
        os.makedirs("config/calibration", exist_ok=True)
        path = f"config/calibration/{self.camera_id}_homography.json"
        self.detector.save_calibration(path)
        print(f"[CAL] ✅ Tersimpan di: {path}")

    def run(self):
        print(f"[CAL] Membuka Stream: {self.video_source}")
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print("[CAL] ❌ Gagal membuka video. Pastikan server Flask sudah jalan!")
            return

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            self.frame = frame
            self.draw_ui()
            
            cv2.imshow(self.window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'): break
            elif key == ord('r'): 
                self.points = []
                self.detector.is_calibrated = False
            elif key == ord('s'): 
                self.save_calibration()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, required=True, help='ID Kamera (CAM-1, dll)')
    parser.add_argument('--video', type=str, help='URL atau Index Kamera')
    args = parser.parse_args()
    
    # Jika --video tidak diisi, otomatis ambil dari stream lokal Flask kamu
    source = args.video if args.video else f"http://localhost:5000/video_feed/{args.camera}"
    
    tool = CalibrationTool(args.camera, source)
    tool.run()