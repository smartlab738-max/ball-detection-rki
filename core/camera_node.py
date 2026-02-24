import socket
import threading
import json
import cv2
import os
import time
import queue
from collections import deque
from datetime import datetime

# Import TurboJPEG Aman
try:
    from turbojpeg import TurboJPEG
    turbo_ok = True
except:
    turbo_ok = False

class CameraNode:
    def __init__(self, node_id, config, ai_processor):
        self.id = node_id
        self.ip = config['ip']
        self.video_port = config.get('video_port', 8080) 
        self.stream_url = f"http://{self.ip}:8080/stream"
        self.label = config['label']
        self.cmd_port = 8000
        
        self.ai = ai_processor
        self.ai_enabled = False 
        
        # === FPS TRACKING ===
        self.fps_timestamps = deque(maxlen=30)  
        self.last_fps_update = time.time()
        self.current_fps = 0.0

        # === ASYNC AI VARIABLES (OPTIMASI 2) ===
        # Antrean dengan ukuran 1 agar aman (Thread-Safe) dan tidak lag
        self.ai_queue = queue.Queue(maxsize=1)
        self.last_ai_frame = None

        # === FITUR VAR (VIDEO ASSISTANT REFEREE) ===
        # Asumsi 15 FPS x 5 Detik = 75 Frame
        self.var_buffer = deque(maxlen=75) 
        self.is_saving_var = False # Flag agar tidak bentrok saat proses simpan

        # Init TurboJPEG (Untuk encode MJPEG yang super cepat ke Web)
        self.jpeg = None
        if turbo_ok:
            try:
                paths = ['C:\\libjpeg-turbo-gcc64\\bin\\turbojpeg.dll', 'turbojpeg.dll']
                for p in paths:
                    if os.path.exists(p):
                        self.jpeg = TurboJPEG(p)
                        break
                if not self.jpeg: self.jpeg = TurboJPEG() 
                print(f"[{self.id}] TurboJPEG Ready.")
            except: 
                print(f"[{self.id}] Fallback to OpenCV.")

        self.current_frame_bytes = None
        self.running = False

    def set_ai_status(self, enabled):
        """Menghidupkan atau mematikan proses AI YOLO untuk kamera ini"""
        self.ai_enabled = enabled
        if not enabled:
            self.last_ai_frame = None 
            # Bersihkan sisa antrean jika AI dimatikan secara tiba-tiba
            while not self.ai_queue.empty():
                try:
                    self.ai_queue.get_nowait()
                except queue.Empty:
                    break
        print(f"[{self.id}] AI Mode: {enabled}")

    def start_receiver(self):
        """Memulai proses pengambilan video dari uStreamer Raspberry Pi"""
        self.running = True
        
        # Kosongkan ingatan VAR saat kamera baru dinyalakan
        self.var_buffer.clear()
        
        # 1. Thread untuk Video Stream (Jalur Cepat)
        self.t_stream = threading.Thread(target=self._stream_loop)
        self.t_stream.daemon = True
        self.t_stream.start()
        
        # 2. Thread untuk AI YOLO (Jalur Lambat)
        self.t_ai = threading.Thread(target=self._ai_worker)
        self.t_ai.daemon = True
        self.t_ai.start()

    def stop_receiver(self):
        """OPTIMASI 3: Mematikan thread dengan aman saat Restart agar CPU tidak panas"""
        self.running = False
        print(f"[{self.id}] Menghentikan service kamera...")
        # Jeda 0.5 detik agar thread Stream & AI sempat membaca status running = False
        time.sleep(0.5)

    def _stream_loop(self):
        """Loop utama pengambil gambar dari Raspberry Pi"""
        print(f"[{self.id}] Menghubungkan ke {self.stream_url}...")
        cap = cv2.VideoCapture(self.stream_url)
        # Buffer size 1 agar kita selalu dapat gambar paling real-time, tidak ada delay
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # === Update FPS ===
            now = time.time()
            self.fps_timestamps.append(now)
            if now - self.last_fps_update >= 0.5:
                self.current_fps = self._calculate_fps()
                self.last_fps_update = now

            # === FITUR VAR: TABUNG GAMBAR ===
            # Kita simpan salinan gambar asli agar replay VAR jernih,
            # bebas dari coretan kotak hijau YOLO.
            self.var_buffer.append(frame.copy())

            # === AI LOGIC (Aman dengan Queue) ===
            if not self.ai_enabled:
                display_frame = frame
            else:
                # Lempar gambar ke antrean AI HANYA jika antrean sedang kosong
                if self.ai_queue.empty():
                    try:
                        self.ai_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass # Jika mendadak penuh (race condition), abaikan saja
                
                # Selalu tampilkan hasil deteksi AI terakhir di layar web
                display_frame = self.last_ai_frame if self.last_ai_frame is not None else frame

            # === Encode Ulang untuk Web Dashboard ===
            if self.jpeg:
                self.current_frame_bytes = self.jpeg.encode(display_frame, quality=60)
            else:
                _, buf = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                self.current_frame_bytes = buf.tobytes()

        cap.release()
        print(f"[{self.id}] Koneksi stream ditutup.")

    def _ai_worker(self):
        """Thread terpisah khusus untuk menjalankan Model AI secara paralel"""
        while self.running:
            if self.ai_enabled:
                try:
                    # Tunggu maksimal 0.05 detik untuk gambar baru di antrean
                    frame_to_process = self.ai_queue.get(timeout=0.05)
                    
                    # Jalankan proses YOLO dan penggambaran garis
                    self.last_ai_frame = self.ai.process(frame_to_process)
                    
                except queue.Empty:
                    # Jika antrean kosong, diam sebentar lalu cek lagi
                    pass
                except Exception as e:
                    print(f"[{self.id}] AI Error: {e}")
            else:
                # Istirahat 50ms jika AI mati agar CPU laptop hemat daya
                time.sleep(0.05)
                
        print(f"[{self.id}] Worker AI dihentikan.")

    # === FUNGSI EKSEKUTOR VAR ===
    def save_var_clip(self):
        """Mengekspor isi buffer (75 gambar terakhir) menjadi video MP4 Slow-Motion"""
        # Cek apakah durasi video sudah cukup panjang (Minimal 1 detik / 15 frame)
        if len(self.var_buffer) < 15:
            return {"success": False, "message": "Video VAR belum cukup panjang (Tunggu beberapa detik setelah Start)"}
            
        if self.is_saving_var:
            return {"success": False, "message": "Sistem sedang memproses klip lain, harap tunggu..."}
            
        self.is_saving_var = True
        try:
            # 1. Snapshot memori gambar saat ini juga
            frames_to_save = list(self.var_buffer)
            
            # 2. Siapkan folder penyimpanan
            output_dir = "var_clips"
            os.makedirs(output_dir, exist_ok=True)
            
            # 3. Beri nama file otomatis (Contoh: VAR_CAM-1_143022.mp4)
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"VAR_{self.id}_{timestamp}.mp4"
            filepath = os.path.join(output_dir, filename)
            
            # 4. Setup Pembuat Video (VideoWriter OpenCV)
            tinggi, lebar, _ = frames_to_save[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Format Codec standar MP4
            
            # Trik SLOW MOTION: Frame direkam 15 FPS, tapi kita putar di kecepatan 5 FPS
            slow_mo_fps = 5.0 
            writer = cv2.VideoWriter(filepath, fourcc, slow_mo_fps, (lebar, tinggi))
            
            # 5. Jahit semua gambar menjadi 1 video utuh
            for f in frames_to_save:
                writer.write(f)
                
            writer.release()
            self.is_saving_var = False
            
            print(f"[VAR] ✅ Klip tersimpan: {filepath}")
            return {"success": True, "message": f"Klip VAR tersimpan: {filename}"}
            
        except Exception as e:
            self.is_saving_var = False
            print(f"[VAR] ❌ Error menyimpan klip: {e}")
            return {"success": False, "message": f"Gagal menyimpan: {e}"}

    # === Helper Methods ===
    def _calculate_fps(self):
        if len(self.fps_timestamps) < 2: return 0.0
        time_diff = self.fps_timestamps[-1] - self.fps_timestamps[0]
        if time_diff == 0: return 0.0
        fps = (len(self.fps_timestamps) - 1) / time_diff
        return round(fps, 2)

    def get_fps(self):
        if len(self.fps_timestamps) < 2: return 0.0
        return self.current_fps

    def send_command(self, action, width, height, fps):
        cmd = {"action": action, "resolution": [width, height], "fps": fps, "port": self.video_port}
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3) # Timeout 3 detik agar tidak menggantung jika Raspi mati
            s.connect((self.ip, self.cmd_port))
            s.send(json.dumps(cmd).encode())
            res = s.recv(1024).decode()
            s.close()
            return res
        except Exception as e:
            return f"Error TCP: {e}"
            
    def get_frame(self):
        return self.current_frame_bytes