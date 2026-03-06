import paho.mqtt.client as mqtt
import threading
import time

class NetSensorListener:
    def __init__(self, system_manager, broker_ip="broker.emqx.io", port=1883):
        self.system_manager = system_manager
        self.broker_ip = broker_ip
        self.port = port
        self.client = mqtt.Client(client_id=f"PingPongWindowsServer_{int(time.time())}")
        
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        self.is_running = False
        self.thread = None

    def start(self):
        """Memulai listener MQTT di thread latar belakang (tidak mengganggu kamera/AI)"""
        self.is_running = True
        self.thread = threading.Thread(target=self._run_mqtt, daemon=True)
        self.thread.start()
        print(f"[MQTT] Memulai pencarian sinyal dari Net Sensor di {self.broker_ip}...")

    def _run_mqtt(self):
        try:
            self.client.connect(self.broker_ip, self.port, 60)

            while self.is_running:
                self.client.loop(timeout=1.0)
        except Exception as e:
            print(f"[MQTT] ❌ Gagal terhubung ke Broker ({self.broker_ip}): {e}")

    def stop(self):
        self.is_running = False
        self.client.disconnect()
        print("[MQTT] Mematikan koneksi ke Net Sensor.")


    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] ✅ Berhasil terhubung ke Broker {self.broker_ip}!")
            self.client.subscribe("esp32/vibration")
            print("[MQTT] Mendengarkan topik: 'esp32/vibration'")
        else:
            print(f"[MQTT] ❌ Koneksi gagal dengan kode: {rc}")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode("utf-8")
        if topic == "esp32/vibration" and (payload == "HIT" or payload == "BERGETAR"):
            print("\n=========================================")
            print("🏐 [ALARM VAR] SENSOR NET BERGETAR! (NET TOUCH)")
            print("=========================================")
            if self.system_manager:
                print("[MQTT] Memerintahkan Auto-VAR untuk momen Net...")
                threading.Thread(target=self.system_manager.save_global_var).start()
                
    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print("[MQTT] ⚠️ Terputus tiba-tiba dari Broker. Mencoba menyambung kembali...")
        else:
            print("[MQTT] Terputus dari Broker secara normal.")