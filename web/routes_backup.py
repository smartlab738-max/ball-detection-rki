from flask import Blueprint, render_template, Response, request, jsonify
import time
import cv2
import numpy as np

bp = Blueprint('main', __name__)
manager = None

def init_routes(app, mgr):
    global manager
    manager = mgr
    app.register_blueprint(bp)

@bp.route('/')
def index():
    return render_template('dashboard.html', nodes=manager.get_all_nodes())

def gen(nid):
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
    return Response(gen(nid), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/api/control', methods=['POST'])
def control():
    tgt = request.form.get('target')
    act = request.form.get('action')
    w = int(request.form.get('width', 512))
    h = int(request.form.get('height', 384))
    fps = int(request.form.get('fps', 15))

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

@bp.route('/api/calibrate/auto/<nid>', methods=['POST'])
def auto_calibrate(nid):
    node = manager.get_node(nid)
    if not node: return jsonify({"success": False, "message": "Node not found"}), 404
    
    frame_bytes = node.get_frame()
    if not frame_bytes: return jsonify({"success": False, "message": "Frame not ready"}), 400
    
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    success = node.ai.auto_calibrate(frame)
    return jsonify({"success": success, "message": "Kalibrasi Berhasil" if success else "Gagal Deteksi Meja"})

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