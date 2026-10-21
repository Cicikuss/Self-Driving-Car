import tensorflow as tf
from tensorflow.keras.models import model_from_json

# 1. Read the network structure from the JSON file
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = tf.keras.models.model_from_json(loaded_model_json)

# 2. Load the trained weights into the reconstructed structure
model_json.save("model_name.h5", save_format="hf")

print("Model successfully loaded from JSON and H5.")
