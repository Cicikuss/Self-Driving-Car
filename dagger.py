import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from pynput import keyboard

class DAgger:
    def __init__(self, data_dir="Data"):
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, "IMG")
        os.makedirs(self.img_dir, exist_ok=True)
        
        self.records = []
        self.recovery_mode = False
        
        self.current_angle = 0.0
        self.throttle = 0.0
        self.pressed_keys = set()
        
        self.steer_step = 0.08
        self.steer_decay = 0.85
        self.throttle_step = 0.05
        
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def generate_time_stamp(self):
        now = datetime.now()
        ms = now.microsecond // 1000
        return now.strftime("%Y_%m_%d_%H_%M_%S_") + f"{ms:03d}"

    def on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char is not None:
                ch = key.char.lower()
                if ch == 'r' and ch not in self.pressed_keys:
                    self.recovery_mode = not self.recovery_mode
                    print(f"\n[DAgger] Recovery Mode: {self.recovery_mode}")
                self.pressed_keys.add(ch)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if hasattr(key, 'char') and key.char is not None:
                ch = key.char.lower()
                self.pressed_keys.discard(ch)
        except AttributeError:
            pass

    def update_controls(self):
        if 'a' in self.pressed_keys and 'd' not in self.pressed_keys:
            self.current_angle -= self.steer_step
        elif 'd' in self.pressed_keys and 'a' not in self.pressed_keys:
            self.current_angle += self.steer_step
        else:
            self.current_angle *= self.steer_decay
            if abs(self.current_angle) < 0.01:
                self.current_angle = 0.0

        if 'w' in self.pressed_keys:
            self.throttle = min(1.0, self.throttle + self.throttle_step)
        elif 's' in self.pressed_keys:
            self.throttle = max(-1.0, self.throttle - self.throttle_step)
        else:
            self.throttle = 0.0

        self.current_angle = float(np.clip(self.current_angle, -1.0, 1.0))
        self.throttle = float(np.clip(self.throttle, -1.0, 1.0))

    def add_data(self, image, target_steering, target_throttle, speed):
        date = self.generate_time_stamp()
        filename = f"center_{date}.jpg"
        abs_save_path = os.path.join(self.img_dir, filename)

        
        self.records.append({
            "center": abs_save_path,
            "left": "none",
            "right": "none",
            "steering": float(target_steering),
            "throttle": float(target_throttle),
            "reverse": 0.0,
            "speed": float(speed)
        })

        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(abs_save_path, img_bgr)
    
    def save_csv(self):
        if not self.records:
            return
        
        csv_path = os.path.join(self.data_dir, "driving_log.csv")
        df = pd.DataFrame(self.records)

        # Tipleri kesinleştir
        df = df.astype({
            "center": str,
            "left": str,
            "right": str,
            "steering": float,
            "throttle": float,
            "reverse": float,
            "speed": float
        })

        # header=False: Udacity standardına uygun şekilde asla başlık satırı yazılmaz
        df.to_csv(csv_path, mode='a', header=False, index=False)
        
        print(f"[DAgger] {len(self.records)} adet veri kaydedildi (headersız): {csv_path}")
        self.records.clear()