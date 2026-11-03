import argparse
import base64
from io import BytesIO
import cv2
import eventlet
from flask import Flask
from keras.models import load_model
import numpy as np
from PIL import Image
import socketio

from dagger import DAgger
from visualize import Visualize

sio = socketio.Server()
app = Flask(__name__)

speed_limit = 10
visualizer = None
dagger = None
model = None

def parse_args():
    parser = argparse.ArgumentParser(prog="drive.py")
    parser.add_argument("-v", "--visualize", action="store_true")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    parser_bc = subparsers.add_parser("behavioral_cloning")
    parser_bc.add_argument("--model", default="model.keras")

    parser_dagger = subparsers.add_parser("dagger")
    parser_dagger.add_argument("--model", default="dagger_model.keras")
    parser_dagger.add_argument("--dagger-data-dir", default="dagger_data")

    return parser.parse_args()

def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    return img / 255.0

@sio.on('telemetry')
def telemetry(sid, data):
    if not data:
        return

    speed = float(data['speed'])
    image_bytes = BytesIO(base64.b64decode(data['image']))
    raw_image = np.asarray(Image.open(image_bytes))

    processed_img = img_preprocess(raw_image)
    batch_img = np.expand_dims(processed_img, axis=0)

    default_throttle = float(np.clip(1.0 - speed / speed_limit, -0.1, 1.0))
    steering_angle = 0.0
    throttle = default_throttle

    if visualizer:
        steering_angle = float(visualizer.visualize(batch_img))

    elif dagger:
        dagger.update_controls()
        model_angle = float(model(batch_img, training=False).numpy()[0][0])

        if dagger.recovery_mode:
            
            steering_angle = dagger.current_angle
            throttle = dagger.throttle if dagger.throttle != 0.0 else default_throttle

            
            dagger.add_data(
                image=raw_image,
                target_steering=steering_angle,
                target_throttle=throttle,
                speed=speed
            )
        else:
            steering_angle = model_angle
            throttle = default_throttle

    else:
        steering_angle = float(model(batch_img, training=False).numpy()[0][0])
        throttle = default_throttle

    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print(f"[Socket] Connected: {sid}")
    send_control(0.0, 0.0)
    if visualizer:
        visualizer.start_window()

@sio.on('disconnect')
def disconnect(sid):
    print(f"[Socket] Disconnected: {sid}")
    if visualizer:
        visualizer.stop()
    if dagger:
        dagger.save_csv()

def send_control(steering_angle, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': str(float(steering_angle)),
            'throttle': str(float(throttle))
        }
    )

def main():
    args = parse_args()
    global model, app, dagger, visualizer

    model = load_model(args.model, compile=False)

    if args.subcommand == "dagger":
        dagger = DAgger(data_dir=args.dagger_data_dir)

    if args.visualize:
        visualizer = Visualize(model)

    app = socketio.Middleware(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    except KeyboardInterrupt:
        print("\n[Server] Interrupted.")
    finally:
        if dagger:
            dagger.save_csv()

if __name__ == '__main__':
    main()