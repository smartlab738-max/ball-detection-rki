"""
Zone Mapper Module - Smart PingPong System
Maps ball positions to table zones and tracks statistics.

Zone Layout (3x3 grid):
┌─────────┬─────────┬─────────┐
│ Zone 0  │ Zone 1  │ Zone 2  │  ← Far (Player A side)
│ BH-Far  │ Mid-Far │ FH-Far  │
├─────────┼─────────┼─────────┤
│ Zone 3  │ Zone 4  │ Zone 5  │  ← Middle
│ BH-Mid  │ Center  │ FH-Mid  │
├─────────┼─────────┼─────────┤
│ Zone 6  │ Zone 7  │ Zone 8  │  ← Near (Player B side)
│ BH-Near │Mid-Near │ FH-Near │
└─────────┴─────────┴─────────┘

BH = Backhand side
FH = Forehand side
"""

import cv2
import numpy as np
import json
from collections import defaultdict


class ZoneMapper:
    """
    Maps pixel coordinates to table zones using homography transformation.
    Tracks hit statistics per zone.
    """
    
    # Zone names (0-8)
    ZONE_NAMES = [
        "BH-Far",  "Mid-Far",  "FH-Far",   # Row 0 (far)
        "BH-Mid",  "Center",   "FH-Mid",   # Row 1 (middle)
        "BH-Near", "Mid-Near", "FH-Near"   # Row 2 (near)
    ]
    
    # Zone colors for visualization (BGR)
    ZONE_COLORS = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),  # Row 0
        (255, 255, 100), (255, 100, 255), (100, 255, 255),  # Row 1
        (200, 200, 100), (200, 100, 200), (100, 200, 200)   # Row 2
    ]
    
    def __init__(self, homography_matrix, table_width=274.0, table_height=152.5):
        """
        Initialize zone mapper.
        
        Args:
            homography_matrix: 3x3 transformation matrix from court_detector
            table_width: Real table width in cm (default: 274.0)
            table_height: Real table height in cm (default: 152.5)
        """
        self.H = homography_matrix
        self.table_width = table_width
        self.table_height = table_height
        
        # Statistics tracking
        self.zone_counts = {i: 0 for i in range(9)}
        self.zone_names = self.ZONE_NAMES
        self.total_detections = 0
        
        # History tracking (optional - for advanced features)
        self.zone_history = []  # List of (zone_id, timestamp)
        
    def pixel_to_real(self, px, py):
        """
        Transform pixel coordinates to real-world coordinates (bird's eye view).
        
        Args:
            px, py: Pixel coordinates in original frame
            
        Returns:
            (x, y) in real-world coordinates (cm from top-left corner)
        """
        # Create point in homogeneous coordinates
        point = np.array([[[px, py]]], dtype=np.float32)
        
        # Apply perspective transform
        transformed = cv2.perspectiveTransform(point, self.H)
        
        real_x, real_y = transformed[0][0]
        return real_x, real_y
    
    def get_zone_id(self, px, py):
        """
        Map pixel coordinate to zone ID (0-8).
        
        Args:
            px, py: Pixel coordinates in original frame
            
        Returns:
            Zone ID (0-8) or -1 if out of bounds
        """
        try:
            # Transform to real coordinates
            real_x, real_y = self.pixel_to_real(px, py)
            
            # Normalize to 0-1 range
            norm_x = real_x / self.table_width
            norm_y = real_y / self.table_height
            
            # Check if out of bounds
            if norm_x < 0 or norm_x > 1 or norm_y < 0 or norm_y > 1:
                return -1  # Out of table
            
            # Clamp to valid range (handle edge cases)
            norm_x = max(0.0, min(0.999, norm_x))
            norm_y = max(0.0, min(0.999, norm_y))
            
            # Map to 3x3 grid
            col = int(norm_x * 3)  # 0, 1, or 2
            row = int(norm_y * 3)  # 0, 1, or 2
            
            zone_id = row * 3 + col
            return zone_id
            
        except Exception as e:
            print(f"[ZONE] Error mapping coordinate: {e}")
            return -1
    
    def update_stats(self, zone_id, timestamp=None):
        """
        Update statistics for a zone.
        
        Args:
            zone_id: Zone ID (0-8)
            timestamp: Optional timestamp (for time-based analysis)
        """
        if zone_id < 0 or zone_id >= 9:
            return  # Invalid zone
        
        self.zone_counts[zone_id] += 1
        self.total_detections += 1
        
        if timestamp is not None:
            self.zone_history.append((zone_id, timestamp))
    
    def get_zone_percentage(self, zone_id):
        """
        Get percentage of hits in a specific zone.
        
        Args:
            zone_id: Zone ID (0-8)
            
        Returns:
            Percentage (0-100)
        """
        if self.total_detections == 0:
            return 0.0
        
        return (self.zone_counts[zone_id] / self.total_detections) * 100.0
    
    def get_heatmap_array(self):
        """
        Get 3x3 array representing zone density.
        
        Returns:
            3x3 numpy array with normalized values (0-1)
        """
        heatmap = np.zeros((3, 3), dtype=np.float32)
        
        if self.total_detections == 0:
            return heatmap
        
        # Fill array
        for zone_id in range(9):
            row = zone_id // 3
            col = zone_id % 3
            heatmap[row, col] = self.zone_counts[zone_id] / self.total_detections
        
        return heatmap
    
    def get_heatmap_normalized(self):
        """
        Get heatmap values normalized to max value (for visualization).
        
        Returns:
            List of 9 values normalized to 0-1 range
        """
        max_count = max(self.zone_counts.values()) if self.total_detections > 0 else 1
        
        return [self.zone_counts[i] / max_count for i in range(9)]
    
    def draw_zone_label(self, frame, px, py, zone_id, 
                       font_scale=0.6, color=(0, 255, 255), thickness=2):
        """
        Draw zone label at ball position.
        
        Args:
            frame: Frame to draw on
            px, py: Ball pixel position
            zone_id: Zone ID
            font_scale: Text size
            color: Text color (BGR)
            thickness: Text thickness
            
        Returns:
            Modified frame
        """
        if zone_id < 0 or zone_id >= 9:
            label = "OUT"
            color = (0, 0, 255)  # Red for out of bounds
        else:
            label = f"Z{zone_id}: {self.zone_names[zone_id]}"
        
        # Draw text background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        bg_pt1 = (int(px) - 5, int(py) - text_size[1] - 30)
        bg_pt2 = (int(px) + text_size[0] + 5, int(py) - 25)
        cv2.rectangle(frame, bg_pt1, bg_pt2, (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, label, 
                   (int(px), int(py) - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)
        
        return frame
    
    def visualize_heatmap(self, frame, corners, alpha=0.5):
        """
        Overlay heatmap visualization on frame.
        
        Args:
            frame: Input frame
            corners: 4 corner points of table
            alpha: Transparency (0=invisible, 1=opaque)
            
        Returns:
            Frame with heatmap overlay
        """
        overlay = frame.copy()
        
        # Get heatmap data
        heatmap = self.get_heatmap_array()
        
        if np.sum(heatmap) == 0:
            return frame  # No data yet
        
        # Draw colored zones
        corners_int = corners.astype(np.int32)
        
        for zone_id in range(9):
            row = zone_id // 3
            col = zone_id % 3
            
            # Calculate zone corners
            # Interpolate based on grid position
            t_vertical = row / 3.0
            b_vertical = (row + 1) / 3.0
            l_horizontal = col / 3.0
            r_horizontal = (col + 1) / 3.0
            
            # Top-left corner of zone
            tl = self._interpolate_point(corners_int, l_horizontal, t_vertical)
            # Top-right corner of zone
            tr = self._interpolate_point(corners_int, r_horizontal, t_vertical)
            # Bottom-right corner of zone
            br = self._interpolate_point(corners_int, r_horizontal, b_vertical)
            # Bottom-left corner of zone
            bl = self._interpolate_point(corners_int, l_horizontal, b_vertical)
            
            zone_corners = np.array([tl, tr, br, bl], dtype=np.int32)
            
            # Get heat intensity
            intensity = heatmap[row, col]
            
            # Color based on intensity (red = hot, blue = cold)
            if intensity > 0:
                # Gradient from blue (cold) to red (hot)
                color_b = int(255 * (1 - intensity))
                color_r = int(255 * intensity)
                color = (color_b, 0, color_r)
                
                # Fill zone
                cv2.fillPoly(overlay, [zone_corners], color)
        
        # Blend with original
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return result
    
    def _interpolate_point(self, corners, h_ratio, v_ratio):
        """
        Interpolate point within quadrilateral.
        
        Args:
            corners: [TL, TR, BR, BL] corners
            h_ratio: Horizontal ratio (0-1, left to right)
            v_ratio: Vertical ratio (0-1, top to bottom)
            
        Returns:
            Interpolated point (x, y)
        """
        # Top edge interpolation
        top = corners[0] * (1 - h_ratio) + corners[1] * h_ratio
        
        # Bottom edge interpolation
        bottom = corners[3] * (1 - h_ratio) + corners[2] * h_ratio
        
        # Vertical interpolation
        point = top * (1 - v_ratio) + bottom * v_ratio
        
        return point.astype(np.int32)
    
    def reset_stats(self):
        """Reset all statistics (e.g., start new match)."""
        self.zone_counts = {i: 0 for i in range(9)}
        self.total_detections = 0
        self.zone_history = []
        print("[ZONE] 🔄 Statistics reset")
    
    def export_stats(self):
        """
        Export statistics as dictionary.
        
        Returns:
            Dictionary with zone stats
        """
        return {
            "zone_counts": self.zone_counts,
            "zone_names": self.zone_names,
            "total_detections": self.total_detections,
            "percentages": {i: self.get_zone_percentage(i) for i in range(9)},
            "heatmap": self.get_heatmap_array().tolist()
        }
    
    def save_stats(self, filepath):
        """
        Save statistics to JSON file.
        
        Args:
            filepath: Path to save JSON
        """
        data = self.export_stats()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[ZONE] 💾 Statistics saved to {filepath}")
    
    @staticmethod
    def load_stats(filepath, homography_matrix):
        """
        Load statistics from JSON file.
        
        Args:
            filepath: Path to JSON file
            homography_matrix: Homography matrix
            
        Returns:
            ZoneMapper instance with loaded stats
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        mapper = ZoneMapper(homography_matrix)
        mapper.zone_counts = data['zone_counts']
        mapper.total_detections = data['total_detections']
        
        print(f"[ZONE] 📂 Statistics loaded from {filepath}")
        return mapper


# === STANDALONE TESTING ===
if __name__ == "__main__":
    """
    Test zone mapper with sample data.
    """
    import matplotlib.pyplot as plt
    
    # Create dummy homography matrix (identity for testing)
    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    mapper = ZoneMapper(H, table_width=274, table_height=152)
    
    print("=== Zone Mapper Test ===")
    
    # Simulate ball detections
    print("\nSimulating ball hits...")
    test_positions = [
        (50, 30),   # Zone 0 (BH-Far)
        (137, 30),  # Zone 1 (Mid-Far)
        (220, 30),  # Zone 2 (FH-Far)
        (137, 76),  # Zone 4 (Center)
        (220, 120), # Zone 8 (FH-Near)
        (137, 76),  # Zone 4 again
        (220, 120), # Zone 8 again
    ]
    
    for px, py in test_positions:
        zone = mapper.get_zone_id(px, py)
        mapper.update_stats(zone)
        print(f"Ball at ({px:3d}, {py:3d}) → Zone {zone} ({mapper.zone_names[zone]})")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total detections: {mapper.total_detections}")
    print("\nZone breakdown:")
    for i in range(9):
        count = mapper.zone_counts[i]
        pct = mapper.get_zone_percentage(i)
        print(f"  Zone {i} ({mapper.zone_names[i]:10s}): {count:2d} hits ({pct:5.1f}%)")
    
    # Visualize heatmap
    print("\nGenerating heatmap visualization...")
    heatmap = mapper.get_heatmap_array()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Hit Density')
    plt.title('Ping Pong Table Heatmap (3x3 Zones)')
    
    # Add zone labels
    for i in range(9):
        row = i // 3
        col = i % 3
        count = mapper.zone_counts[i]
        plt.text(col, row, f'{mapper.zone_names[i]}\n{count}', 
                ha='center', va='center', color='white', fontsize=10)
    
    plt.xticks([0, 1, 2], ['Backhand', 'Middle', 'Forehand'])
    plt.yticks([0, 1, 2], ['Far', 'Middle', 'Near'])
    plt.tight_layout()
    plt.savefig('/home/claude/zone_heatmap_test.png', dpi=150)
    print("Heatmap saved to: zone_heatmap_test.png")
    
    # Export stats
    mapper.save_stats('/home/claude/zone_stats_test.json')