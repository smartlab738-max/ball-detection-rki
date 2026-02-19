# import socket
# import threading
# import json
# import cv2
# import numpy as np
# import os

# # Import TurboJPEG Aman
# try:
#     from turbojpeg import TurboJPEG
#     turbo_ok = True
# except:
#     turbo_ok = False

# class CameraNode:
#     def __init__(self, node_id, config, ai_processor):
#         self.id = node_id
#         self.ip = config['ip']
#         self.video_port = config['video_port']
#         self.label = config['label']
#         self.cmd_port = 8000
        
#         self.ai = ai_processor
#         self.ai_enabled = False # Default AI MATI (Mode Ringan)
        
#         # Init TurboJPEG
#         self.jpeg = None
#         if turbo_ok:
#             try:
#                 # Cek Path Default Windows
#                 paths = ['C:\\libjpeg-turbo-gcc64\\bin\\turbojpeg.dll', 'turbojpeg.dll']
#                 for p in paths:
#                     if os.path.exists(p):
#                         self.jpeg = TurboJPEG(p)
#                         break
#                 if not self.jpeg: self.jpeg = TurboJPEG() # Try System Path
#                 print(f"[{self.id}] TurboJPEG Ready.")
#             except: 
#                 print(f"[{self.id}] Fallback to OpenCV.")

#         self.current_frame_bytes = None
#         self.running = False
        
#         # Var Frame Skipping
#         self.frame_count = 0

#     def set_ai_status(self, enabled):
#         self.ai_enabled = enabled
#         print(f"[{self.id}] AI Mode: {enabled}")

#     def start_receiver(self):
#         self.running = True
#         t = threading.Thread(target=self._udp_loop)
#         t.daemon = True
#         t.start()

#     def _udp_loop(self):
#         sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         sock.bind(('0.0.0.0', self.video_port))
#         sock.settimeout(0.01) # Timeout Cepat

#         print(f"[{self.id}] Listening Video...")

#         while self.running:
#             try:
#                 # 1. BUFFER FLUSHING (Hapus antrian lama, ambil yang terbaru)
#                 data = None
#                 while True:
#                     try:
#                         chunk, _ = sock.recvfrom(65535)
#                         data = chunk
#                     except socket.timeout:
#                         break
                
#                 if data is None: continue

#                 # 2. Validasi & Decode
#                 if data[:2] == b'\xff\xd8':
#                     frame = None
#                     if self.jpeg:
#                         frame = self.jpeg.decode(data)
#                     else:
#                         np_arr = np.frombuffer(data, np.uint8)
#                         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#                     if frame is not None:
#                         # 3. AI LOGIC (Hanya jika saklar ON)
#                         if self.ai_enabled:
#                             # Frame Skipping: Proses AI cuma tiap 3 frame sekali
#                             self.frame_count += 1
#                             if self.frame_count % 3 == 0:
#                                 frame = self.ai.process(frame)
#                             # Frame lainnya lewat tanpa kotak (atau pakai kotak lama jika mau kompleks)
                        
#                         # 4. Encode Ulang untuk Web
#                         if self.jpeg:
#                             self.current_frame_bytes = self.jpeg.encode(frame, quality=60)
#                         else:
#                             _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
#                             self.current_frame_bytes = buf.tobytes()

#             except Exception:
#                 pass
#         sock.close()

#     def send_command(self, action, width, height, fps):
#         cmd = {"action": action, "resolution": [width, height], "fps": fps, "port": self.video_port}
#         try:
#             s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             s.settimeout(3)
#             s.connect((self.ip, self.cmd_port))
#             s.send(json.dumps(cmd).encode())
#             res = s.recv(1024).decode()
#             s.close()
#             return res
#         except Exception as e:
#             return f"Error: {e}"
            
#     def get_frame(self):
#         return self.current_frame_bytes


import socket
import threading
import json
import cv2
import numpy as np
import os
import time
from collections import deque

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
        self.video_port = config['video_port']
        self.label = config['label']
        self.cmd_port = 8000
        
        self.ai = ai_processor
        self.ai_enabled = False # Default AI MATI (Mode Ringan)
        
        # === TAMBAHAN: FPS TRACKING ===
        self.fps_timestamps = deque(maxlen=30)  # Menyimpan 30 frame terakhir
        self.last_fps_update = time.time()
        self.current_fps = 0.0

        # Init TurboJPEG
        self.jpeg = None
        if turbo_ok:
            try:
                # Cek Path Default Windows
                paths = ['C:\\libjpeg-turbo-gcc64\\bin\\turbojpeg.dll', 'turbojpeg.dll']
                for p in paths:
                    if os.path.exists(p):
                        self.jpeg = TurboJPEG(p)
                        break
                if not self.jpeg: self.jpeg = TurboJPEG() # Try System Path
                print(f"[{self.id}] TurboJPEG Ready.")
            except: 
                print(f"[{self.id}] Fallback to OpenCV.")

        self.current_frame_bytes = None
        self.running = False
        
        # Var Frame Skipping
        self.frame_count = 0

    def set_ai_status(self, enabled):
        self.ai_enabled = enabled
        print(f"[{self.id}] AI Mode: {enabled}")

    def start_receiver(self):
        self.running = True
        t = threading.Thread(target=self._udp_loop)
        t.daemon = True
        t.start()

    def _udp_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.video_port))
        sock.settimeout(0.01) # Timeout Cepat

        print(f"[{self.id}] Listening Video...")

        while self.running:
            try:
                # 1. BUFFER FLUSHING
                data = None
                while True:
                    try:
                        chunk, _ = sock.recvfrom(65535)
                        data = chunk
                    except socket.timeout:
                        break
                
                if data is None: continue

                # 2. Validasi & Decode
                if data[:2] == b'\xff\xd8':
                    frame = None
                    if self.jpeg:
                        frame = self.jpeg.decode(data)
                    else:
                        np_arr = np.frombuffer(data, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        # === TAMBAHAN: Track FPS ===
                        now = time.time()
                        self.fps_timestamps.append(now)
                        
                        # Update nilai FPS setiap 0.5 detik agar stabil
                        if now - self.last_fps_update >= 0.5:
                            self.current_fps = self._calculate_fps()
                            self.last_fps_update = now

                        # 3. AI LOGIC
                        if self.ai_enabled:
                            self.frame_count += 1
                            if self.frame_count % 3 == 0:
                                frame = self.ai.process(frame)
                        
                        # 4. Encode Ulang untuk Web
                        if self.jpeg:
                            self.current_frame_bytes = self.jpeg.encode(frame, quality=60)
                        else:
                            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                            self.current_frame_bytes = buf.tobytes()

            except Exception:
                pass
        sock.close()

    # === TAMBAHAN: Helper Methods untuk FPS ===
    def _calculate_fps(self):
        """Menghitung FPS berdasarkan timestamp dalam deque."""
        if len(self.fps_timestamps) < 2:
            return 0.0
        
        time_diff = self.fps_timestamps[-1] - self.fps_timestamps[0]
        if time_diff == 0:
            return 0.0
        
        fps = (len(self.fps_timestamps) - 1) / time_diff
        return round(fps, 2)

    def get_fps(self):
        """Mengambil nilai FPS saat ini untuk API."""
        if len(self.fps_timestamps) < 2:
            return 0.0
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