"""
FPS Diagnostic Tool - Smart PingPong
Measures actual FPS at each stage of the pipeline to identify bottleneck.

Usage:
    python diagnose_fps.py
"""

import cv2
import time
import socket
import numpy as np
from collections import deque
import sys


class FPSDiagnostic:
    """Measure FPS at different pipeline stages."""
    
    def __init__(self, window_size=30):
        self.window = window_size
        self.timestamps = {
            'udp_receive': deque(maxlen=window_size),
            'jpeg_decode': deque(maxlen=window_size),
            'display': deque(maxlen=window_size)
        }
    
    def mark(self, stage):
        """Mark timestamp for a stage."""
        self.timestamps[stage].append(time.time())
    
    def get_fps(self, stage):
        """Calculate FPS for a stage."""
        ts = self.timestamps[stage]
        if len(ts) < 2:
            return 0.0
        
        time_diff = ts[-1] - ts[0]
        if time_diff == 0:
            return 0.0
        
        return (len(ts) - 1) / time_diff
    
    def print_report(self):
        """Print FPS report."""
        print("\n" + "="*60)
        print("FPS DIAGNOSTIC REPORT")
        print("="*60)
        
        for stage in ['udp_receive', 'jpeg_decode', 'display']:
            fps = self.get_fps(stage)
            status = "✅ GOOD" if fps >= 8 else "⚠️ SLOW"
            print(f"{stage:20s}: {fps:6.2f} FPS  {status}")
        
        print("="*60)
        
        # Diagnosis
        rx_fps = self.get_fps('udp_receive')
        decode_fps = self.get_fps('jpeg_decode')
        display_fps = self.get_fps('display')
        
        print("\n📊 DIAGNOSIS:")
        if rx_fps < 8:
            print("❌ NETWORK SLOW - Check WiFi signal, try lower quality")
        elif decode_fps < rx_fps - 2:
            print("❌ DECODE SLOW - Install TurboJPEG or use lower resolution")
        elif display_fps < decode_fps - 2:
            print("❌ DISPLAY SLOW - Check browser/Flask performance")
        else:
            print("✅ All stages GOOD!")
        
        print("")


def test_udp_receiver(port=9991, duration=10):
    """Test UDP packet receiving speed."""
    print(f"\n🔍 TEST 1: UDP Receiver Performance")
    print(f"Port: {port}")
    print(f"Duration: {duration} seconds")
    print("-" * 60)
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', port))
        sock.settimeout(1.0)
    except Exception as e:
        print(f"❌ ERROR: Cannot bind to port {port}")
        print(f"   {e}")
        print(f"   Make sure server is NOT running!")
        return
    
    diag = FPSDiagnostic()
    frame_count = 0
    start_time = time.time()
    last_report = start_time
    
    total_bytes = 0
    
    print("Receiving packets... (waiting for data)")
    
    try:
        while time.time() - start_time < duration:
            try:
                data, addr = sock.recvfrom(65535)
                diag.mark('udp_receive')
                frame_count += 1
                total_bytes += len(data)
                
                # Print stats every 2 seconds
                if time.time() - last_report >= 2.0:
                    fps = diag.get_fps('udp_receive')
                    avg_size = total_bytes / frame_count if frame_count > 0 else 0
                    mbps = (total_bytes * 8 / 1000000) / (time.time() - start_time)
                    
                    print(f"📊 Packets: {frame_count:4d} | FPS: {fps:5.1f} | "
                          f"Avg Size: {avg_size/1000:.1f}KB | {mbps:.2f} Mbps")
                    
                    last_report = time.time()
                    
            except socket.timeout:
                if frame_count == 0:
                    print("⏳ Waiting for packets... (is camera streaming?)")
                continue
                
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    
    sock.close()
    
    print("\n" + "="*60)
    print("TEST 1 RESULTS:")
    print("="*60)
    print(f"Total packets: {frame_count}")
    print(f"Average FPS: {diag.get_fps('udp_receive'):.1f}")
    if frame_count > 0:
        print(f"Average packet size: {total_bytes/frame_count/1000:.1f} KB")
        print(f"Total bandwidth: {(total_bytes*8/1000000)/(time.time()-start_time):.2f} Mbps")
    
    if diag.get_fps('udp_receive') < 8:
        print("\n⚠️  WARNING: Network receiving is SLOW!")
        print("   Try: Lower quality/resolution on Raspberry Pi")
    else:
        print("\n✅ Network receiving is GOOD!")


def test_jpeg_decode_speed(port=9991, duration=10):
    """Test JPEG decoding speed."""
    print(f"\n🔍 TEST 2: JPEG Decode Performance")
    print(f"Port: {port}")
    print(f"Duration: {duration} seconds")
    print("-" * 60)
    
    # Try importing TurboJPEG
    try:
        from turbojpeg import TurboJPEG
        jpeg_decoder = TurboJPEG()
        print("✅ Using TurboJPEG (fast)")
        use_turbo = True
    except:
        jpeg_decoder = None
        print("⚠️  Using OpenCV imdecode (slower)")
        print("   Install TurboJPEG for better performance: pip install PyTurboJPEG")
        use_turbo = False
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', port))
        sock.settimeout(1.0)
    except Exception as e:
        print(f"❌ ERROR: Cannot bind to port {port}")
        print(f"   {e}")
        return
    
    diag = FPSDiagnostic()
    frame_count = 0
    decode_count = 0
    start_time = time.time()
    last_report = start_time
    
    print("\nReceiving and decoding... (waiting for data)")
    
    try:
        while time.time() - start_time < duration:
            try:
                data, addr = sock.recvfrom(65535)
                diag.mark('udp_receive')
                
                # Decode JPEG
                decode_start = time.time()
                
                if use_turbo and jpeg_decoder:
                    try:
                        frame = jpeg_decoder.decode(data)
                        decode_count += 1
                    except:
                        frame = None
                else:
                    nparr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        decode_count += 1
                
                decode_time = (time.time() - decode_start) * 1000  # ms
                
                diag.mark('jpeg_decode')
                frame_count += 1
                
                # Print stats every 2 seconds
                if time.time() - last_report >= 2.0:
                    rx_fps = diag.get_fps('udp_receive')
                    dec_fps = diag.get_fps('jpeg_decode')
                    
                    print(f"📊 RX: {rx_fps:5.1f} FPS | Decode: {dec_fps:5.1f} FPS | "
                          f"Decode time: {decode_time:.1f}ms")
                    
                    if frame is not None:
                        print(f"   Frame shape: {frame.shape}")
                    
                    last_report = time.time()
                    
            except socket.timeout:
                if frame_count == 0:
                    print("⏳ Waiting for packets...")
                continue
                
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    
    sock.close()
    
    print("\n" + "="*60)
    print("TEST 2 RESULTS:")
    print("="*60)
    print(f"Total packets: {frame_count}")
    print(f"Decoded frames: {decode_count}")
    print(f"RX FPS: {diag.get_fps('udp_receive'):.1f}")
    print(f"Decode FPS: {diag.get_fps('jpeg_decode'):.1f}")
    
    if diag.get_fps('jpeg_decode') < diag.get_fps('udp_receive') - 2:
        print("\n⚠️  WARNING: Decoding is SLOWER than receiving!")
        if not use_turbo:
            print("   Try: pip install PyTurboJPEG")
        else:
            print("   Try: Lower resolution on Raspberry Pi")
    else:
        print("\n✅ Decoding performance is GOOD!")


def test_full_pipeline(port=9991, duration=30):
    """Test complete pipeline with visualization."""
    print(f"\n🔍 TEST 3: Full Pipeline (End-to-End)")
    print(f"Port: {port}")
    print(f"Duration: {duration} seconds")
    print("-" * 60)
    
    # Try importing TurboJPEG
    try:
        from turbojpeg import TurboJPEG
        jpeg_decoder = TurboJPEG()
        print("✅ Using TurboJPEG")
        use_turbo = True
    except:
        jpeg_decoder = None
        print("⚠️  Using OpenCV")
        use_turbo = False
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', port))
        sock.settimeout(0.1)
    except Exception as e:
        print(f"❌ ERROR: Cannot bind to port {port}")
        print(f"   {e}")
        return
    
    diag = FPSDiagnostic()
    
    window_name = f"FPS Test - Port {port}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("\nDisplaying video...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' for stats")
    print("-" * 60)
    
    last_report = time.time()
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            try:
                # Receive UDP
                data, addr = sock.recvfrom(65535)
                diag.mark('udp_receive')
                
                # Decode JPEG
                if use_turbo and jpeg_decoder:
                    try:
                        frame = jpeg_decoder.decode(data)
                    except:
                        frame = None
                else:
                    nparr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                diag.mark('jpeg_decode')
                
                if frame is not None:
                    frame_count += 1
                    
                    # Add FPS overlay
                    fps_udp = diag.get_fps('udp_receive')
                    fps_decode = diag.get_fps('jpeg_decode')
                    fps_display = diag.get_fps('display')
                    
                    # Background for text
                    cv2.rectangle(frame, (5, 5), (300, 110), (0, 0, 0), -1)
                    
                    # FPS text
                    cv2.putText(frame, f"UDP RX:  {fps_udp:5.1f} FPS", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Decode:  {fps_decode:5.1f} FPS", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Display: {fps_display:5.1f} FPS", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 255), 2)
                    
                    # Show frame
                    cv2.imshow(window_name, frame)
                    diag.mark('display')
                    
                    # Print report every 5 seconds
                    if time.time() - last_report >= 5.0:
                        diag.print_report()
                        last_report = time.time()
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    diag.print_report()
                    
            except socket.timeout:
                if frame_count == 0:
                    # Show waiting screen
                    waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(waiting, "Waiting for video stream...", 
                               (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.0, (255, 255, 255), 2)
                    cv2.imshow(window_name, waiting)
                    cv2.waitKey(1)
                continue
                
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    
    sock.close()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("TEST 3 FINAL RESULTS:")
    print("="*60)
    diag.print_report()
    print(f"\nTotal frames displayed: {frame_count}")
    print(f"Test duration: {time.time()-start_time:.1f} seconds")


def main():
    """Main diagnostic menu."""
    print("""
╔════════════════════════════════════════════════╗
║   FPS DIAGNOSTIC TOOL - Smart PingPong         ║
╚════════════════════════════════════════════════╝

⚠️  IMPORTANT: Stop the main server before running!
   (Otherwise port will be in use)

Select test:
  1. Test UDP receive only (network speed)
  2. Test UDP + JPEG decode (processing speed)
  3. Test full pipeline with display (end-to-end) ⭐ RECOMMENDED
  
  q. Quit
    """)
    
    choice = input("Enter choice [1-3]: ").strip()
    
    if choice == '1':
        port_str = input("Enter UDP port [default: 9991]: ").strip()
        port = int(port_str) if port_str else 9991
        duration_str = input("Test duration in seconds [default: 10]: ").strip()
        duration = int(duration_str) if duration_str else 10
        test_udp_receiver(port, duration)
        
    elif choice == '2':
        port_str = input("Enter UDP port [default: 9991]: ").strip()
        port = int(port_str) if port_str else 9991
        duration_str = input("Test duration in seconds [default: 10]: ").strip()
        duration = int(duration_str) if duration_str else 10
        test_jpeg_decode_speed(port, duration)
        
    elif choice == '3':
        port_str = input("Enter UDP port [default: 9991]: ").strip()
        port = int(port_str) if port_str else 9991
        duration_str = input("Test duration in seconds [default: 30]: ").strip()
        duration = int(duration_str) if duration_str else 30
        
        print("\n📝 NOTES:")
        print("   - Video window will open")
        print("   - FPS stats overlaid on video")
        print("   - Report printed every 5 seconds")
        print("   - Press 'q' to quit early")
        print("   - Press 's' for stats anytime")
        
        input("\nPress ENTER to start...")
        
        test_full_pipeline(port, duration)
        
    elif choice.lower() == 'q':
        print("👋 Exiting...")
        sys.exit(0)
    else:
        print("❌ Invalid choice!")
        sys.exit(1)
    
    # Ask to run another test
    print("\n" + "="*60)
    again = input("\nRun another test? [y/N]: ").strip().lower()
    if again == 'y':
        main()
    else:
        print("👋 Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
        sys.exit(0)