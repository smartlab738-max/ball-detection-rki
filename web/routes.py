from flask import Blueprint, render_template, Response, request, jsonify
import time
import cv2
import numpy as np
import json
import os

bp = Blueprint('main', __name__)
manager = None

def init_routes(app, mgr):
    global manager
    manager = mgr
    app.register_blueprint(bp)

@bp.route('/')
def index():
    # Menampilkan halaman utama dashboard
    return render_template('dashboard.html', nodes=manager.get_all_nodes())

def gen(nid):
    """
    Generator untuk stream video. 
    Mengambil frame berupa bytes JPEG dari CameraNode.
    """
    node = manager.get_node(nid)
    while True:
        frame = node.get_frame() if node else None
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)
        time.sleep(0.01)

@bp.route('/video_feed/<nid>')
def video_feed(nid):
    """Endpoint URL untuk tag <img> di HTML"""
    return Response(gen(nid), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/api/control', methods=['POST'])
def control():
    """
    Endpoint untuk tombol Start/Stop Kamera.
    Akan meneruskan perintah ke port 8000 di Raspberry Pi (cmd_listener.py)
    """
    tgt = request.form.get('target')
    act = request.form.get('action')
    
    # Beri nilai default 0 karena uStreamer sudah mengunci resolusi dan FPS
    w = int(request.form.get('width', 0))
    h = int(request.form.get('height', 0))
    fps = int(request.form.get('fps', 0))

    if tgt == 'ALL': 
        res = manager.broadcast_command(act, w, h, fps)
    else: 
        res = manager.single_command(tgt, act, w, h, fps)
        
    return jsonify(res)

@bp.route('/api/toggle_ai', methods=['POST'])
def toggle_ai():
    tgt = request.form.get('target', 'ALL')
    status = request.form.get('status') == 'true'
    res = manager.set_ai_mode(tgt, status)
    return jsonify(res)

# === DIPERBARUI: MENGGUNAKAN GAMBAR BERSIH DARI VAR BUFFER ===
@bp.route('/api/calibrate/auto/<nid>', methods=['POST'])
def auto_calibrate(nid):
    node = manager.get_node(nid)
    if not node: return jsonify({"success": False, "message": "Node not found"}), 404
    
    if not node.running:
         return jsonify({"success": False, "message": "Kamera sedang mati. Nyalakan dulu (START)."}), 400
    
    # Ambil gambar bersih dari memori VAR agar coretan AI tidak mengganggu deteksi garis
    if hasattr(node, 'var_buffer') and len(node.var_buffer) > 0:
        frame = node.var_buffer[-1].copy()
    else:
        # Fallback jika memori VAR belum siap
        frame_bytes = node.get_frame()
        if not frame_bytes: return jsonify({"success": False, "message": "Frame not ready"}), 400
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    success = node.ai.auto_calibrate(frame)
    return jsonify({
        "success": success, 
        "message": "Kalibrasi Otomatis Berhasil!" if success else "Gagal Deteksi Meja. Coba Manual."
    })

@bp.route('/api/analytics/<nid>')
def get_analytics(nid):
    node = manager.get_node(nid)
    if not node or not node.ai: return jsonify({"error": "No AI"}), 404
    
    stats = node.ai.get_zone_stats()
    # Fix KeyError dengan default value
    if not stats:
        return jsonify({"calibrated": False, "total_hits": 0, "zones": [0]*9, "zone_names": ["Z"]*9})

    return jsonify({
        "camera_id": nid,
        "calibrated": True,
        "zones": [stats['zone_counts'].get(str(i), 0) for i in range(9)],
        "zone_names": stats['zone_names'],
        "total_hits": stats['total_detections']
    })

@bp.route('/api/analytics/reset/<nid>', methods=['POST'])
def reset_analytics(nid):
    node = manager.get_node(nid)
    if node and node.ai:
        node.ai.reset_zone_stats()
        return jsonify({"success": True})
    return jsonify({"success": False}), 404

@bp.route('/api/court/toggle/<nid>', methods=['POST'])
def toggle_court_overlay(nid):
    node = manager.get_node(nid)
    if node and node.ai:
        enabled = request.form.get('enabled') == 'true'
        node.ai.toggle_court_overlay(enabled)
        return jsonify({"success": True})
    return jsonify({"success": False}), 404

@bp.route('/api/zone_grid/toggle/<nid>', methods=['POST'])
def toggle_zone_grid(nid):
    node = manager.get_node(nid)
    if node and node.ai:
        enabled = request.form.get('enabled') == 'true'
        node.ai.toggle_zone_grid(enabled)
        return jsonify({"success": True})
    return jsonify({"success": False}), 404

@bp.route('/api/calibrate', methods=['POST'])
def calibrate_camera():
    """
    Web-based calibration endpoint.
    Receives 4 corner points from browser and computes homography.
    """
    camera_id = request.form.get('camera_id')
    corners_json = request.form.get('corners')
    
    if not camera_id or not corners_json:
        return jsonify({"success": False, "error": "Missing parameters"}), 400
    
    try:
        # Parse corners from JSON
        corners_raw = json.loads(corners_json)
        
        # Convert to numpy array (format: [[x,y], [x,y], [x,y], [x,y]])
        corners = np.array([[pt['x'], pt['y']] for pt in corners_raw], dtype=np.float32)
        
        # Import court detector
        from core.court_detector import CourtDetector
        
        # Create detector and set corners
        detector = CourtDetector()
        detector.set_manual_corners(corners)
        
        # Compute homography
        detector.compute_homography(scale_factor=2.0)
        
        # Save calibration
        cal_dir = "config/calibration"
        os.makedirs(cal_dir, exist_ok=True)
        
        cal_path = os.path.join(cal_dir, f"{camera_id}_homography.json")
        detector.save_calibration(cal_path)
        
        # Reload calibration in the active camera node
        node = manager.get_node(camera_id)
        if node and node.ai:
            # Reload AI processor calibration (Ini akan memanggil _update_drawing_cache otomatis)
            node.ai._load_calibration()
        
        return jsonify({
            "success": True,
            "camera_id": camera_id,
            "message": "Calibration saved successfully"
        })
    
    except Exception as e:
        print(f"[API] Calibration error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route('/api/calibration_status/<nid>')
def get_calibration_status(nid):
    """
    Check if a camera is calibrated.
    """
    cal_path = f"config/calibration/{nid}_homography.json"
    calibrated = os.path.exists(cal_path)
    
    return jsonify({
        "camera_id": nid,
        "calibrated": calibrated,
        "path": cal_path if calibrated else None
    })

# === FPS MONITORING ENDPOINT ===
@bp.route('/api/fps/<nid>')
def get_fps(nid):
    """
    Get current FPS for a camera.
    """
    node = manager.get_node(nid)
    
    if not node:
        return jsonify({"camera_id": nid, "fps": None, "error": "Camera not found"}), 404
    
    # Get FPS from camera node
    fps = node.get_fps() if hasattr(node, 'get_fps') else None
    
    return jsonify({
        "camera_id": nid,
        "fps": fps,
        "timestamp": time.time()
    })

# === FITUR VAR ENDPOINT (BARU) ===
@bp.route('/api/var/save/<nid>', methods=['POST'])
def save_var(nid):
    """Endpoint untuk mengekspor isi Ring Buffer menjadi video MP4 Slow-Motion"""
    node = manager.get_node(nid)
    
    if not node:
        return jsonify({"success": False, "message": "Kamera tidak ditemukan"}), 404
    
    if not node.running:
        return jsonify({"success": False, "message": "Kamera sedang mati, memori video kosong!"}), 400
        
    # Panggil fungsi penjahit MP4 di dalam camera_node.py
    if hasattr(node, 'save_var_clip'):
        result = node.save_var_clip()
        return jsonify(result)
    else:
        return jsonify({"success": False, "message": "Fitur VAR belum tersedia di node ini"}), 501
    
@bp.route('/api/var/list')
def list_var_clips():
    """Mengambil daftar semua video klip VAR yang tersimpan"""
    var_dir = "var_clips"
    if not os.path.exists(var_dir):
        return jsonify([])
    # Ambil file mp4 dan urutkan (yang terbaru di atas)
    files = [f for f in os.listdir(var_dir) if f.endswith('.mp4')]
    files.sort(reverse=True) 
    return jsonify(files)

@bp.route('/var_clips/<filename>')
def serve_var_clip(filename):
    """Menyajikan file video MP4 ke web browser"""
    return send_from_directory(os.path.abspath("var_clips"), filename)