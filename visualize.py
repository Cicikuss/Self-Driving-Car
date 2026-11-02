import cv2
from keras import models
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Lambda, MaxPooling2D
from pynput import keyboard
import numpy as np


class Visualize:
    def __init__(self,model,cols=8,tile = 96,per_channel_norm=True,cmap = cv2.COLORMAP_VIRIDIS):
        self.model = model
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        self.cols = cols
        self.tile = tile
        self.per_channel_norm = per_channel_norm
        self.cmap = cmap
        self.show_kernels = False
        self.page =0
        self.layer_names = [layer.name for layer in self.model.layers if isinstance(layer, Conv2D) or isinstance(layer, MaxPooling2D)]
        self.layer_outputs = [layer.output for layer in self.model.layers if isinstance(layer, Conv2D) or isinstance(layer, MaxPooling2D)]
        self.layer_outputs.append(self.model.outputs[-1]) 
        self.activation_model = Model(inputs=self.model.inputs, outputs=self.layer_outputs)
        
    
    def start_window(self):
        cv2.namedWindow("Activation", cv2.WINDOW_NORMAL)

    def stop(self):
        cv2.destroyAllWindows()
        self.listener.stop()


    def on_press(self, key):
        n = len(self.layer_names)
        try:
            if key == keyboard.Key.right:
                self.page = (self.page + 1) % n
            elif key == keyboard.Key.left:
                self.page = (self.page - 1) % n
            elif hasattr(key,'char') and key.char:
                c = key.char.lower()
                if c == 'k':
                    self.show_kernels = not self.show_kernels
                elif c == 'q':
                    self.per_channel_norm = not self.per_channel_norm
        except AttributeError:
            pass
        

    def concat_vh(self,list_2d):
        return cv2.vconcat([cv2.hconcat(list_h) 
                            for list_h in list_2d])

    #-----------Helper functions for visualization----------------


    def kernel_grid(self,page):
        pass

    def activation_grid(self,acts):
        pass



    #-----------Main function for visualization----------------
    def visualize(self, image):
        #Thread müdahalesi olmaması için page ve show_kernels değişkenlerini local değişkenlere atadım.
        #Bir nevi snapshot almış oldum. Böylece threadler arası değişken çakışması olmayacak.
        page , show_kernels = self.page, self.show_kernels

        #Predict yerine bunu kullanmak daha mantıklıymış çünkü predict batch hazırlığı yapıyor
        #Bunda direkt forward pass yapılıyor ve tensorler döndürülüyor.Opencv de kullanmak için numpy arraye çevirmek gerekiyor.
        acts = [np.asarray(act) for act in self.activation_model(image, training=False)] 
        steering_angle = float(acts[-1][0][0])
        
        grid = self.kernel_grid(page) if show_kernels else None
        if grid is not None:
            title = f"Kernels: {self.layer_names[page]}"
        else:
            act = acts[page][0]
            grid = self.activation_grid(act)
            title = f"{self.layer_names[page]} {act.shape}"

        inp = cv2.normalize(image[0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if inp.ndim == 2 or inp.shape[-1] == 1:
        
        cv2.waitKey(1)

        return steering_angle
        
        

        
                