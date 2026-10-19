import cv2
from keras import models
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Lambda, MaxPooling2D
from pynput import keyboard


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
        cv2.namedWindow("Activation", cv2.WINDOW_GUI_EXPANDED)

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


    def visualize(self, image):
        activations = self.activation_model.predict(image)
        
        steering_angle = float(activations[-1][0][0])
        concataned_image = []
        for i in range (activations[self.page].shape[-1]):   
            concataned_image.append(cv2.resize(activations[self.page][0,:,:,i], (64, 64)))
        cv2.imshow("Activation",cv2.hconcat(concataned_image))
        cv2.waitKey(1) 
       

        return steering_angle
        
        

        
                