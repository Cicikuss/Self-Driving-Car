
import cv2
from keras import models
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Lambda, MaxPooling2D
from pynput import keyboard
import numpy as np


class Visualize:
    #Şeyi kabul edicem Claude normalization ve grid yapmayı benden daha iyi yaptı.Benim versiyonumdan daha iyi oldu.
    def __init__(self,model,cols=8,tile = 96,per_channel_norm=True,cmap = cv2.COLORMAP_VIRIDIS):
        self.model = model
        self.cols = cols
        self.tile = tile
        self.per_channel_norm = per_channel_norm
        self.cmap = cmap
        self.page = 0
        self.show_kernels = False

        self.layers = [layer for layer in model.layers
                       if isinstance(layer, (Conv2D, MaxPooling2D))]
        self.layer_names = [layer.name for layer in self.layers]
        outputs = [layer.output for layer in self.layers] + [model.outputs[-1]]
        self.activation_model = Model(model.inputs, outputs)

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        
    
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
        

    

    #-----------Helper functions for visualization----------------
    def _normalize(self,maps):
        maps = maps.astype(np.float32)
        if self.per_channel_norm:
            mn = maps.min(axis=(0,1),keepdims=True)
            mx = maps.max(axis=(0,1),keepdims=True)
        else:
            mn = maps.min()
            mx = maps.max()
        return ((maps - mn) / (mx - mn + 1e-8)*255).astype(np.uint8)

    def _make_grid(self,tiles,colorize=True):
        tile = self.tile
        out = []
        for i ,img in enumerate(tiles):
            img = cv2.resize(img,(tile,tile),interpolation=cv2.INTER_NEAREST)
            if img.ndim == 2:
                img = cv2.applyColorMap(img,self.cmap) if colorize else cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            cv2.putText(img, str(i), (3, 12), cv2.FONT_HERSHEY_PLAIN,0.9, (255, 255, 255), 1)
            img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            out.append(img)
        blank =np.zeros_like(out[0])
        while len(out) % self.cols:
            out.append(blank)
        rows = [np.hstack(out[i:i+self.cols]) for i in range(0, len(out), self.cols)]
        return np.vstack(rows)


    def activation_grid(self,activation):
            maps = self._normalize(activation)
            return self._make_grid(np.transpose(maps, (2, 0, 1)))
    
    


    def kernel_grid(self,page):
        layer = self.layers[page]
        if not isinstance(layer, Conv2D):
            return None
        weights = layer.get_weights()[0]
        kernel_height, kernel_width, in_channels, out_channels = weights.shape

        tiles = []
        for out in range(out_channels):
            kernel = weights[:,:,:,out]
            kernel = ((kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8) * 255).astype(np.uint8)
            tiles.append(kernel[:, :, ::-1] if in_channels == 3 else kernel.mean(axis=2).astype(np.uint8))
        return self._make_grid(tiles, colorize=(in_channels != 3))

    


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
            #Resmin tek kanallı yani grayscale olduğunda garantiye almak için renkli hale getiriyorum. 
            # Çünkü cv2.imshow tek kanallı resimleri göstermiyor.
            inp = cv2.cvtColor(inp.squeeze(), cv2.COLOR_GRAY2BGR)

        
        inp = cv2.resize(inp, (160, int(160 * inp.shape[0] / inp.shape[1])))
        header = np.zeros((max(inp.shape[0], 80), grid.shape[1], 3), np.uint8)
        header[:inp.shape[0],:inp.shape[1]] = inp
        x0 = inp.shape[1] + 10
        cv2.putText(header,title,(x0,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        cv2.putText(header,f"Steering Angle: {steering_angle:+.3f}",(x0,48),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        cv2.putText(header, "<- ->: layer   k: kernels   n: norm mode", (x0, 72),cv2.FONT_HERSHEY_PLAIN, 1, (180, 180, 180), 1)


        cv2.imshow("Activation", np.vstack([header, grid]))
        cv2.waitKey(1)
        
        

        return steering_angle
        
        

        
                