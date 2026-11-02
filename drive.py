from flask import Flask
import eventlet
import socketio
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from visualize import Visualize
import argparse
from dagger import DAgger

sio =socketio.Server()
app = Flask(__name__)
speed_limit = 10
visualizer = None
dagger = None
model = None

def parse_args():
    parser = argparse.ArgumentParser(prog="drive.py",description="Run the imitation learning model in a simulator.")
    parser.add_argument("--v","--visualize", action="store_true", help="Visualize the model's activations.")
    subparsers = parser.add_subparsers(dest="subcommand",help = "Sub commands for different techniques and model types.")
    subparsers.required = True    
    parser_behavioral_cloning = subparsers.add_parser("behavioral_cloning", help="Run the behavioral cloning model.")
    parser_behavioral_cloning.add_argument("--model", default="model.keras", help="Path to the trained model file.")

    parser_dagger = subparsers.add_parser("dagger", help="Run the DAgger model.")
    parser_dagger.add_argument("--model", default="dagger_model.keras", help="Path to the trained DAgger model file.")
    parser_dagger.add_argument("--dagger-data-dir", default="dagger_data", help="Directory containing DAgger data.")


    return parser.parse_args()

def img_preprocess(img):
  img = img[60:135,:,:]
  img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img,(3,3),0)
  img = cv2.resize(img,(200,66))
  img = img/255
  return img




@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    throttle = 1.0 - speed/speed_limit
    steering_angle = 0.0
    if visualizer:
        steering_angle = visualizer.visualize(image)
    elif dagger:
        steering_angle = dagger.current_angle
    else:
        steering_angle = model(image, training=False).numpy()[0][0]
    
    
    
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
    
    

@sio.on('connect')
def connect(sid,environ):
    print('connect ', sid)
    send_control(0,0)
    if visualizer:
        visualizer.start_window()

@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)
    if visualizer:
        visualizer.close_window()
    if dagger:
        dagger.save_csv()

def send_control(steering_angle,throttle):
    sio.emit('steer', data={'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()  })


def main():
    args = parse_args()
    global model
    global app
    global dagger
    global visualizer
    if args.subcommand == "behavioral_cloning":
        model = load_model(args.model, compile=False)
    elif args.subcommand == "dagger":
        model = load_model(args.model, compile=False)
        dagger = DAgger(args.dagger_data_dir)
    if args.v:
        visualizer = Visualize(model)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
        
if __name__ == '__main__':
    #model = load_model('model.h5',compile=False)
    #visualizer = Visualize(model)
    #app = socketio.Middleware(sio,app)
    #eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    main()
    
    



    