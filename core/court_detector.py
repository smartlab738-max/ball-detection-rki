"""
Court Detector Module - Smart PingPong System
Detects ping pong table boundaries and performs perspective transformation.

Features:
- Auto-detect table corners using edge detection
- Manual corner selection (fallback)
- Compute homography matrix for perspective warp
- Overlay court lines on video frames
"""

import cv2
import numpy as np
import json


class CourtDetector:
    """
    Detects ping pong table court and provides perspective transformation.
    """
    
    # Standard ping pong table dimensions (in cm)
    TABLE_WIDTH = 274.0  # cm
    TABLE_HEIGHT = 152.5  # cm
    
    def __init__(self):
        """Initialize court detector with empty state."""
        self.corners = None  # 4 corner points [top-left, top-right, bottom-right, bottom-left]
        self.matrix = None   # 3x3 homography matrix
        self.is_calibrated = False
        
        # === OPTIMASI 1: Variabel Cache untuk Menggambar (Mencegah RAM Leak) ===
        self._cache_court_poly = None
        self._cache_net_line = None
        self._cache_grid_lines = None
        self._cache_corner_markers = None
        
    def _update_drawing_cache(self):
        """
        [OPTIMASI 1 DITERAPKAN DI SINI]
        Menghitung semua matematika dan array HANYA SATU KALI saat kalibrasi.
        Ini menghemat ribuan komputasi per detik.
        """
        if self.corners is None:
            return
            
        corners_int = self.corners.astype(np.int32)
        
        # 1. Cache untuk Garis Tepi Lapangan
        self._cache_court_poly = corners_int.reshape((-1, 1, 2))
        
        # 2. Cache untuk Garis Net
        top_mid = tuple(((corners_int[0] + corners_int[1]) / 2).astype(int))
        bottom_mid = tuple(((corners_int[2] + corners_int[3]) / 2).astype(int))
        self._cache_net_line = (top_mid, bottom_mid)
        
        # 3. Cache untuk Marker Angka Sudut
        self._cache_corner_markers = [tuple(pt) for pt in corners_int]
        
        # 4. Cache untuk Grid Zona (3x3)
        self._cache_grid_lines = []
        
        # -- Hitung Garis Vertikal (divide width into 3)
        for i in range(1, 3):
            t = i / 3.0
            left_pt = tuple((corners_int[0] * (1-t) + corners_int[3] * t).astype(int))
            right_pt = tuple((corners_int[1] * (1-t) + corners_int[2] * t).astype(int))
            self._cache_grid_lines.append((left_pt, right_pt))
            
        # -- Hitung Garis Horizontal (divide height into 3)
        for i in range(1, 3):
            t = i / 3.0
            top_pt = tuple((corners_int[0] * (1-t) + corners_int[1] * t).astype(int))
            bottom_pt = tuple((corners_int[3] * (1-t) + corners_int[2] * t).astype(int))
            self._cache_grid_lines.append((top_pt, bottom_pt))
        
    def auto_detect_corners(self, frame, debug=False):
        # ... (Kode algoritma tidak diubah) ...
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        if debug:
            cv2.imshow("Edges", edges)
            cv2.waitKey(0)
            
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for cnt in contours[:10]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                frame_area = frame.shape[0] * frame.shape[1]
                
                if area > frame_area * 0.1:
                    self.corners = approx.reshape(4, 2).astype(np.float32)
                    self.corners = self._order_corners(self.corners)
                    self._update_drawing_cache() # Update cache setelah dapat koordinat
                    print(f"[COURT] Auto-detected table: {area:.0f} pixels")
                    return self.corners
                    
        print("[COURT] ⚠️ Auto-detection failed - use manual mode")
        return None
    
    def set_manual_corners(self, points):
        self.corners = np.array(points, dtype=np.float32)
        self.corners = self._order_corners(self.corners)
        self._update_drawing_cache() # Update cache
        print(f"[COURT] Manual corners set: {self.corners.tolist()}")
    
    def _order_corners(self, pts):
        pts = pts[np.argsort(pts[:, 1])]
        top = pts[:2]
        top = top[np.argsort(top[:, 0])]
        bottom = pts[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    
    def compute_homography(self, scale_factor=1.0):
        if self.corners is None:
            raise ValueError("Corners not set! Call auto_detect_corners() or set_manual_corners() first")
            
        width = int(self.TABLE_WIDTH * scale_factor)
        height = int(self.TABLE_HEIGHT * scale_factor)
        
        dst = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        self.matrix = cv2.getPerspectiveTransform(self.corners, dst)
        self.is_calibrated = True
        print(f"[COURT] ✅ Homography computed: {width}x{height} output")
        return self.matrix
    
    def warp_frame(self, frame, output_size=None):
        if self.matrix is None:
            raise ValueError("Matrix not computed!")
        if output_size is None:
            output_size = (int(self.TABLE_WIDTH), int(self.TABLE_HEIGHT))
        return cv2.warpPerspective(frame, self.matrix, output_size)
    
    def overlay_court_lines(self, frame, color=(0, 255, 0), thickness=2):
        if self.corners is None or self._cache_court_poly is None:
            return frame 
            
        # === SUPER RINGAN: Langsung pakai array yang sudah dicache ===
        cv2.polylines(frame, [self._cache_court_poly], isClosed=True, color=color, thickness=thickness)
        
        # Gambar Garis Net
        net_p1, net_p2 = self._cache_net_line
        cv2.line(frame, net_p1, net_p2, color, thickness)
        
        # Marker Sudut
        for i, corner in enumerate(self._cache_corner_markers):
            cv2.circle(frame, corner, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i+1), (corner[0] - 10, corner[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def overlay_zone_grid(self, frame, alpha=0.3):
        if self.corners is None or self._cache_grid_lines is None:
            return frame
        
        overlay = frame.copy()
        
        # === SUPER RINGAN: Tidak ada lagi numpy math di dalam loop ini ===
        for line_start, line_end in self._cache_grid_lines:
            cv2.line(overlay, line_start, line_end, (255, 255, 0), 1)
        
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return result
    
    def save_calibration(self, filepath):
        if self.corners is None or self.matrix is None:
            raise ValueError("No calibration to save")
        
        data = {
            "corners": self.corners.tolist(),
            "matrix": self.matrix.tolist(),
            "table_width": self.TABLE_WIDTH,
            "table_height": self.TABLE_HEIGHT
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[COURT] 💾 Calibration saved to {filepath}")
    
    @staticmethod
    def load_calibration(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        detector = CourtDetector()
        detector.corners = np.array(data['corners'], dtype=np.float32)
        detector.matrix = np.array(data['matrix'], dtype=np.float32)
        detector.is_calibrated = True
        
        # WAJIB PANGGIL INI SAAT LOAD
        detector._update_drawing_cache()
        
        print(f"[COURT] 📂 Calibration loaded from {filepath}")
        return detector


# === STANDALONE TESTING ===
if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)
    detector = CourtDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        if detector.is_calibrated:
            display = detector.overlay_court_lines(display)
            display = detector.overlay_zone_grid(display, alpha=0.2)
            
        cv2.imshow("Court Detector Test", display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            corners = detector.auto_detect_corners(frame, debug=True)
            if corners is not None:
                detector.compute_homography(scale_factor=2.0)
        elif key == ord('s'):
            if detector.is_calibrated:
                detector.save_calibration("test_calibration.json")
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()