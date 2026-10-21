import pandas as pd
from pynput import keyboard
import os
import cv2
from datetime import datetime
import numpy as np

class DAgger:
    def __init__(self):
        self.current_dir = os.getcwd()
        self.records = []
        try:
            os.mkdir("IMG")
            print("Directory 'IMG' created successfully.")
        except FileExistsError:
            print("Directory 'IMG' already exists.")

        self.current_angle = 0.0
        columns = ["center","left","right","steering","throttle","reverse","speed"]
        
        self.listener = keyboard.Listener(on_press=self.on_press,on_release=self.on_release)
        self.listener.start()

    def generate_time_stamp(self):
        now = datetime.now()
        ms = now.microsecond // 1000
        return now.strftime("%Y_%m_%d_%H_%M_%S_") + f"{ms:03d}"

    def on_release(self, key):
        self.current_angle = 0.0
    
    def on_press(self, key):
        try:
            match key.char:
                case 'a':
                    self.current_angle -= 0.1
                    
                case 'd':
                    self.current_angle += 0.1
            self.current_angle = np.clip(self.current_angle, -1.0, 1.0)
        except AttributeError:
            pass
    
    def add_data(self,image,throttle,reverse,speed):
        date = self.generate_time_stamp()
        save_path = os.path.join(self.current_dir, "IMG", f"center_{date}.jpg")
        self.records.append((save_path,"left", "right", self.current_angle,throttle,reverse,speed))
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path,img)
        return save_path, self.current_angle, throttle, reverse, speed
    
    def save_csv(self):
        csv_path = os.path.join(self.current_dir, "driving_log.csv")
        pd.DataFrame(self.records, columns=["center","left","right","steering","throttle","reverse","speed"]).to_csv(csv_path, index=False)


    
        
