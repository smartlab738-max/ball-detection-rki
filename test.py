import cv2
import socket
import numpy as np
import threading
import time
from ultralytics import YOLO

# --- KONFIGURASI ---
PORTS = [9991, 9992, 9993, 9994] # Port untuk 4 kamera
LABELS = ["Front View", "Back View", "Left Side", "Right Side"]
DASHBOARD_SIZE = (1280, 720)     # Ukuran jendela admin

# Load Model (Gunakan Nano agar PC tidak meledak memproses 4 cam)
# Jika PC sangat kuat (RTX 3060 ke atas), boleh pakai model custom Anda
model = YOLO('yolov8n.pt') 

# --- CLASS PENERIMA STREAM ---
class CamStream(threading.Thread):
    def __init__(self, port, id):
        threading.Thread.__init__(self)
        self.port = port
        self.id = id
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8) # Frame hitam kosong (default)
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port))
        self.sock.settimeout(0.2) # Timeout biar thread tidak macet selamanya
        
        # Tambahkan teks "NO SIGNAL" di frame default
        cv2.putText(self.frame, "NO SIGNAL", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    def run(self):
        print(f"[INFO] Listening on Port {self.port}...")
        while self.running:
            try:
                data, _ = self.sock.recvfrom(65535)
                # Decode gambar
                np_arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    self.frame = img
            except socket.timeout:
                pass # Tidak ada data, biarkan frame terakhir atau hitam
            except Exception as e:
                print(f"Error Cam {self.id}: {e}")

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.sock.close()

# --- MAIN DASHBOARD LOGIC ---
streams = []
# 1. Start 4 Thread Penerima
for i, port in enumerate(PORTS):
    cam = CamStream(port, i)
    cam.start()
    streams.append(cam)

print("Dashboard Admin Berjalan... Tekan 'q' untuk keluar.")

try:
    while True:
        frames_to_show = []
        
        # 2. Ambil frame dari setiap kamera
        for i, stream in enumerate(streams):
            img = stream.get_frame()
            
            # --- OPSI: JALANKAN YOLO DI SINI ---
            # Peringatan: Menjalankan YOLO 4x per loop akan sangat berat!
            # Solusi: Jalankan YOLO hanya di Cam Utama (misal Cam 0), atau gunakan model Nano.
            # Di sini saya contohkan detect di SEMUA kamera tapi pakai conf rendah biar cepat
            
            results = model.predict(img, conf=0.5, verbose=False, imgsz=320)
            img_annotated = results[0].plot()
            
            # Tambahkan Label Kamera
            cv2.putText(img_annotated, LABELS[i], (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            frames_to_show.append(img_annotated)

        # 3. GABUNGKAN GAMBAR (GRID 2x2)
        # Baris Atas: Cam 1 + Cam 2
        top_row = np.hstack((frames_to_show[0], frames_to_show[1]))
        # Baris Bawah: Cam 3 + Cam 4
        bot_row = np.hstack((frames_to_show[2], frames_to_show[3]))
        
        # Gabung Atas + Bawah
        dashboard = np.vstack((top_row, bot_row))
        
        # Resize agar pas di layar monitor
        dashboard_final = cv2.resize(dashboard, DASHBOARD_SIZE)

        # 4. Tampilkan
        cv2.imshow("CENTRAL ADMIN DASHBOARD - 4 CAMS", dashboard_final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# Bersihkan Thread saat keluar
for stream in streams:
    stream.stop()
cv2.destroyAllWindows()