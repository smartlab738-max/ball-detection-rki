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
        
        self.fps_timestamps = deque(maxlen=30)  
        self.last_fps_update = time.time()
        self.current_fps = 0.0

        self.ai_queue = queue.Queue(maxsize=1)
        self.last_ai_frame = None

        self.var_buffer = deque(maxlen=75) 
        self.is_saving_var = False 

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
        self.ai_enabled = enabled
        if not enabled:
            self.last_ai_frame = None 
            while not self.ai_queue.empty():
                try:
                    self.ai_queue.get_nowait()
                except queue.Empty:
                    break
        print(f"[{self.id}] AI Mode: {enabled}")

    def start_receiver(self):
        self.running = True
        self.var_buffer.clear()
        
        self.t_stream = threading.Thread(target=self._stream_loop)
        self.t_stream.daemon = True
        self.t_stream.start()
        
        self.t_ai = threading.Thread(target=self._ai_worker)
        self.t_ai.daemon = True
        self.t_ai.start()

    def stop_receiver(self):
        self.running = False
        print(f"[{self.id}] Menghentikan service kamera...")
        time.sleep(0.5)

    def _stream_loop(self):
        print(f"[{self.id}] Menghubungkan ke {self.stream_url}...")
        cap = cv2.VideoCapture(self.stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()
            self.fps_timestamps.append(now)
            if now - self.last_fps_update >= 0.5:
                self.current_fps = self._calculate_fps()
                self.last_fps_update = now

            self.var_buffer.append(frame.copy())

            if not self.ai_enabled:
                display_frame = frame
            else:
                if self.ai_queue.empty():
                    try:
                        self.ai_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass 
                display_frame = self.last_ai_frame if self.last_ai_frame is not None else frame

            if self.jpeg:
                self.current_frame_bytes = self.jpeg.encode(display_frame, quality=60)
            else:
                _, buf = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                self.current_frame_bytes = buf.tobytes()

        cap.release()
        print(f"[{self.id}] Koneksi stream ditutup.")

    def _ai_worker(self):
        while self.running:
            if self.ai_enabled:
                try:
                    frame_to_process = self.ai_queue.get(timeout=0.05)
                    self.last_ai_frame = self.ai.process(frame_to_process)
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"[{self.id}] AI Error: {e}")
            else:
                time.sleep(0.05)
        print(f"[{self.id}] Worker AI dihentikan.")

    # === DIPERBARUI: Menerima Waktu Kejadian Bersama (sync_timestamp) ===
    def save_var_clip(self, sync_timestamp=None):
        if len(self.var_buffer) < 15:
            return {"success": False, "message": "Video belum cukup panjang (Tunggu bbrp detik)"}
            
        if self.is_saving_var:
            return {"success": False, "message": "Sedang menyimpan klip lain..."}
            
        self.is_saving_var = True
        try:
            frames_to_save = list(self.var_buffer)
            output_dir = "var_clips"
            os.makedirs(output_dir, exist_ok=True)
            
            # Jika ada waktu bersama (Master VAR), gunakan itu. Jika manual per kamera, buat waktu sendiri.
            timestamp = sync_timestamp if sync_timestamp else datetime.now().strftime("%H%M%S")
            # Format nama: VAR_Jam_IDKamera.mp4 (Agar rapi berurutan di folder)
            filename = f"VAR_{timestamp}_{self.id}.mp4"
            filepath = os.path.join(output_dir, filename)
            
            tinggi, lebar, _ = frames_to_save[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            slow_mo_fps = 5.0 
            writer = cv2.VideoWriter(filepath, fourcc, slow_mo_fps, (lebar, tinggi))
            
            for f in frames_to_save:
                writer.write(f)
                
            writer.release()
            self.is_saving_var = False
            
            print(f"[VAR] ✅ Klip tersimpan: {filepath}")
            return {"success": True, "message": f"VAR Disimpan: {filename}"}
            
        except Exception as e:
            self.is_saving_var = False
            print(f"[VAR] ❌ Error menyimpan: {e}")
            return {"success": False, "message": f"Gagal menyimpan: {e}"}

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
            s.settimeout(3)
            s.connect((self.ip, self.cmd_port))
            s.send(json.dumps(cmd).encode())
            res = s.recv(1024).decode()
            s.close()
            return res
        except Exception as e:
            return f"Error: {e}"
            
    def get_frame(self):
        return self.current_frame_bytes