from ultralytics import YOLO
import os
import cv2

class AIProcessor:
    def __init__(self, model_path):
        # Cek apakah file model ada
        if not os.path.exists(model_path):
            print(f"[AI] ERROR: Model tidak ditemukan di {model_path}")
            print("[AI] Mode Deteksi Non-Aktif.")
            self.model = None
        else:
            print(f"[AI] Memuat Model YOLO: {model_path}...")
            try:
                self.model = YOLO(model_path)
                print("[AI] Model Berhasil Dimuat!")
            except Exception as e:
                print(f"[AI] Error saat load model: {e}")
                self.model = None

    def process(self, frame):
        """
        Input: Frame Gambar (Polos)
        Output: Frame Gambar dengan Overlay Bawaan YOLO
        """
        # Jika model gagal load, kembalikan gambar asli tanpa edit
        if self.model is None:
            return frame

        try:
            # 1. Jalankan Inference (Deteksi)
            # conf=0.45: Hanya deteksi jika yakin > 45%
            # iou=0.5: Hapus kotak yang tumpang tindih
            # verbose=False: Agar terminal tidak penuh tulisan log
            results = self.model(frame, conf=0.35, iou=0.5, verbose=False)
            
            # 2. Ambil Gambar Hasil Plotting (Bawaan YOLO)
            # Fungsi .plot() ini otomatis menggambar kotak, label, dan skor
            annotated_frame = results[0].plot()
            
            return annotated_frame
            
        except Exception as e:
            print(f"[AI] Error Process: {e}")
            return frame # Jika error, kembalikan gambar asli