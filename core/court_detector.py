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
    
    The detector can work in two modes:
    1. Auto-detection: Uses edge detection to find table corners
    2. Manual mode: User provides 4 corner points
    """
    
    # Standard ping pong table dimensions (in cm)
    TABLE_WIDTH = 274.0  # cm
    TABLE_HEIGHT = 152.5  # cm
    
    def __init__(self):
        """Initialize court detector with empty state."""
        self.corners = None  # 4 corner points [top-left, top-right, bottom-right, bottom-left]
        self.matrix = None   # 3x3 homography matrix
        self.is_calibrated = False
        
    def auto_detect_corners(self, frame, debug=False):
        """
        Automatically detect table corners using edge detection.
        
        Args:
            frame: Input BGR image
            debug: If True, show intermediate steps
            
        Returns:
            numpy array of 4 corners or None if detection failed
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        if debug:
            cv2.imshow("Edges", edges)
            cv2.waitKey(0)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Try to find rectangular contour (table should be largest rectangle)
        for cnt in contours[:10]:  # Check top 10 largest contours
            # Approximate contour to polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # If we found a quadrilateral (4 corners)
            if len(approx) == 4:
                # Check if area is reasonable (not too small)
                area = cv2.contourArea(approx)
                frame_area = frame.shape[0] * frame.shape[1]
                
                if area > frame_area * 0.1:  # At least 10% of frame
                    self.corners = approx.reshape(4, 2).astype(np.float32)
                    self.corners = self._order_corners(self.corners)
                    print(f"[COURT] Auto-detected table: {area:.0f} pixels ({area/frame_area*100:.1f}% of frame)")
                    return self.corners
        
        print("[COURT] ⚠️ Auto-detection failed - use manual mode")
        return None
    
    def set_manual_corners(self, points):
        """
        Set corners manually (from user click or predefined).
        
        Args:
            points: List of 4 points [(x,y), (x,y), (x,y), (x,y)]
                   Order: top-left, top-right, bottom-right, bottom-left
        """
        self.corners = np.array(points, dtype=np.float32)
        self.corners = self._order_corners(self.corners)
        print(f"[COURT] Manual corners set: {self.corners.tolist()}")
    
    def _order_corners(self, pts):
        """
        Order points in consistent order: [top-left, top-right, bottom-right, bottom-left]
        
        Args:
            pts: 4 points in any order
            
        Returns:
            Ordered points
        """
        # Sort by y-coordinate (top points first)
        pts = pts[np.argsort(pts[:, 1])]
        
        # Top 2 points
        top = pts[:2]
        top = top[np.argsort(top[:, 0])]  # Sort by x (left first)
        
        # Bottom 2 points
        bottom = pts[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]  # Sort by x (left first)
        
        # Return in order: TL, TR, BR, BL
        ordered = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
        return ordered
    
    def compute_homography(self, scale_factor=1.0):
        """
        Compute homography matrix for perspective transformation.
        
        Args:
            scale_factor: Scale output size (1.0 = use table dimensions in cm)
            
        Returns:
            3x3 homography matrix
        """
        if self.corners is None:
            raise ValueError("Corners not set! Call auto_detect_corners() or set_manual_corners() first")
        
        # Destination points (rectangle with actual table proportions)
        width = int(self.TABLE_WIDTH * scale_factor)
        height = int(self.TABLE_HEIGHT * scale_factor)
        
        dst = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # Compute perspective transform matrix
        self.matrix = cv2.getPerspectiveTransform(self.corners, dst)
        self.is_calibrated = True
        
        print(f"[COURT] ✅ Homography computed: {width}x{height} output")
        return self.matrix
    
    def warp_frame(self, frame, output_size=None):
        """
        Apply perspective warp to get bird's eye view.
        
        Args:
            frame: Input frame
            output_size: (width, height) or None to use table dimensions
            
        Returns:
            Warped frame
        """
        if self.matrix is None:
            raise ValueError("Matrix not computed! Call compute_homography() first")
        
        if output_size is None:
            output_size = (int(self.TABLE_WIDTH), int(self.TABLE_HEIGHT))
        
        warped = cv2.warpPerspective(frame, self.matrix, output_size)
        return warped
    
    def overlay_court_lines(self, frame, color=(0, 255, 0), thickness=2):
        """
        Draw court boundary lines on the original frame.
        
        Args:
            frame: Input frame (will be modified)
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with overlay
        """
        if self.corners is None:
            return frame  # No corners, return original
        
        # Draw table boundary (quadrilateral)
        corners_int = self.corners.astype(np.int32)
        cv2.polylines(frame, [corners_int], isClosed=True, color=color, thickness=thickness)
        
        # Draw center line (net)
        # Net is at half-height of the table
        top_mid = ((corners_int[0] + corners_int[1]) / 2).astype(np.int32)
        bottom_mid = ((corners_int[2] + corners_int[3]) / 2).astype(np.int32)
        cv2.line(frame, tuple(top_mid), tuple(bottom_mid), color, thickness)
        
        # Optional: Draw corner markers
        for i, corner in enumerate(corners_int):
            cv2.circle(frame, tuple(corner), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i+1), tuple(corner - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def overlay_zone_grid(self, frame, alpha=0.3):
        """
        Draw 3x3 zone grid overlay on the frame.
        
        Args:
            frame: Input frame
            alpha: Transparency (0=invisible, 1=opaque)
            
        Returns:
            Frame with zone grid overlay
        """
        if self.corners is None:
            return frame
        
        overlay = frame.copy()
        corners_int = self.corners.astype(np.int32)
        
        # Calculate grid lines (divide into 3x3)
        # Vertical lines (divide width into 3)
        for i in range(1, 3):
            t = i / 3.0
            # Left edge interpolation
            left_pt = (corners_int[0] * (1-t) + corners_int[3] * t).astype(np.int32)
            # Right edge interpolation
            right_pt = (corners_int[1] * (1-t) + corners_int[2] * t).astype(np.int32)
            cv2.line(overlay, tuple(left_pt), tuple(right_pt), (255, 255, 0), 1)
        
        # Horizontal lines (divide height into 3)
        for i in range(1, 3):
            t = i / 3.0
            # Top edge interpolation
            top_pt = (corners_int[0] * (1-t) + corners_int[1] * t).astype(np.int32)
            # Bottom edge interpolation
            bottom_pt = (corners_int[3] * (1-t) + corners_int[2] * t).astype(np.int32)
            cv2.line(overlay, tuple(top_pt), tuple(bottom_pt), (255, 255, 0), 1)
        
        # Blend with original frame
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return result
    
    def save_calibration(self, filepath):
        """
        Save calibration to JSON file.
        
        Args:
            filepath: Path to save JSON
        """
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
        """
        Load calibration from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            CourtDetector instance with loaded calibration
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        detector = CourtDetector()
        detector.corners = np.array(data['corners'], dtype=np.float32)
        detector.matrix = np.array(data['matrix'], dtype=np.float32)
        detector.is_calibrated = True
        
        print(f"[COURT] 📂 Calibration loaded from {filepath}")
        return detector


# === STANDALONE TESTING ===
if __name__ == "__main__":
    """
    Test court detector with webcam or video file.
    
    Usage:
        python court_detector.py                    # Use webcam
        python court_detector.py video.mp4          # Use video file
    """
    import sys
    
    # Video source
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)
    
    detector = CourtDetector()
    
    print("=== Court Detector Test ===")
    print("Press 'a' to auto-detect")
    print("Press 's' to save calibration")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # If calibrated, show overlay
        if detector.is_calibrated:
            display = detector.overlay_court_lines(display)
            display = detector.overlay_zone_grid(display, alpha=0.2)
        
        cv2.imshow("Court Detector Test", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            print("Attempting auto-detection...")
            corners = detector.auto_detect_corners(frame, debug=True)
            if corners is not None:
                detector.compute_homography(scale_factor=2.0)
                
        elif key == ord('s'):
            if detector.is_calibrated:
                detector.save_calibration("test_calibration.json")
            else:
                print("Not calibrated yet!")
                
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()