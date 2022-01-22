import os
import numpy as np
from tensorflow.keras.models import load_model
from src.utils.common import read_config
def get_model(config_path):
    config = read_config(config_path)
    artifact_dir = config['artifacts']['artifact_dir']
    model_dir = config['artifacts']['model_dir']
    model_path = os.path.join(artifact_dir, model_dir)
    files = os.listdir(model_path)
    for i in files:
        lenth = len(i)
        if i[(lenth-3):] == '.h5':
            return model_path +"/" +i
            break
        else:
            pass


def predict(data, config_path):
    model_file = get_model(config_path)
    print(model_file)
    model = load_model(model_file)
    y_pred = model.predict(data)
    print(np.argmax(y_pred, axis=-1))
