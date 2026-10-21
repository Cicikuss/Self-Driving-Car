import cv2
from keras import models
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Lambda, MaxPooling2D
from pynput import keyboard
import numpy as np


class Visualize:
    def __init__(self,model):
        self.model = model
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        self.page =0
        self.layer_names = [layer.name for layer in self.model.layers if isinstance(layer, Conv2D) or isinstance(layer, MaxPooling2D)]
        self.layer_outputs = [layer.output for layer in self.model.layers if isinstance(layer, Conv2D) or isinstance(layer, MaxPooling2D)]
        self.layer_outputs.append(self.model.outputs[-1]) 
        self.activation_model = Model(inputs=self.model.inputs, outputs=self.layer_outputs)
        
    
    def start_window(self):
        cv2.namedWindow("Activation", cv2.WINDOW_NORMAL)


    def on_press(self, key):
        try:
            if key == keyboard.Key.right:
                self.next_page()
            elif key == keyboard.Key.left:
                self.prev_page()
        except AttributeError:
            pass
        
    def next_page(self):
        print("Next Layer")
        self.page += 1
        if self.page >= len(self.layer_names):
            self.page = 0
    def prev_page(self):
        print("Previous Layer")
        self.page -= 1
        if self.page < 0:
            self.page = len(self.layer_names) - 1

    def concat_vh(self,list_2d):
        return cv2.vconcat([cv2.hconcat(list_h) 
                            for list_h in list_2d])


    def visualize(self, image):
        activations = self.activation_model.predict(image)
        steering_angle = float(activations[-1][0][0])
        
        current_activation = activations[self.page][0]
        num_channels = current_activation.shape[-1]
        
        concataned_image = []
        row = []
        cols = 4  
        
        for i in range(num_channels):
            
            resized_feature = cv2.resize(current_activation[:, :, i], (128, 128))
            row.append(resized_feature)
            
            
            if len(row) == cols:
                concataned_image.append(row)
                row = []
                
        
        if len(row) > 0:
            while len(row) < cols:
                row.append(np.zeros((128, 128), dtype=row[0].dtype))
            concataned_image.append(row)
            
        final_grid = self.concat_vh(concataned_image)
        final_grid = cv2.normalize(final_grid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.putText(final_grid, f"Layer: {self.layer_names[self.page]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Activation", final_grid)
        cv2.waitKey(1)

        return steering_angle
        
        

        
                