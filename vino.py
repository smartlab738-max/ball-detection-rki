import cv2
import time
from ultralytics import YOLO

def main():
    # 1. Tentukan path ke FOLDER model OpenVINO Anda.
    # Pastikan Anda sudah menjalankan perintah "yolo export ..." sebelumnya.
    # Contoh path: 'models/best_openvino_model' atau 'yolov8n_openvino_model'
    model_path = 'models/best_openvino_model' 
    
    print(f"[*] Memuat model OpenVINO dari: {model_path}")
    print("[*] Mohon tunggu sebentar, OpenVINO sedang melakukan kompilasi model ke CPU Intel...")
    
    # Load model menggunakan Ultralytics (otomatis mendeteksi format OpenVINO)
    model = YOLO(model_path)
    print("[*] Model berhasil dimuat!")

    # 2. Buka kamera lokal PC (index 0 biasanya untuk webcam utama)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[!] Gagal membuka kamera PC.")
        return

    # Variabel untuk menghitung FPS
    prev_time = 0

    print("[*] Memulai live stream. Tekan 'q' pada keyboard untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Gagal mengambil gambar dari kamera.")
            break

        # 3. Jalankan Inference (Deteksi) dengan OpenVINO
        # verbose=False agar terminal tidak banjir tulisan log
        results = model(frame, verbose=False)

        # 4. Ambil gambar yang sudah digambar kotak (bounding box) bawaan YOLO
        annotated_frame = results[0].plot()

        # 5. Hitung FPS (Frames Per Second)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # 6. Tempelkan teks FPS di pojok kiri atas gambar
        # Format teks: Hijau, ukuran huruf 1, ketebalan 2
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 7. Tampilkan hasil akhirnya ke layar
        cv2.imshow("Test OpenVINO - PC Server", annotated_frame)

        # 8. Keluar dari loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersihkan memory setelah selesai
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Program selesai.")

if __name__ == "__main__":
    main()