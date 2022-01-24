from src.utils.common import read_config
from src.utils.model import get_unique_name
import os
import numpy as np

from src.utils.model import get_unique_name
import tensorflow as tf

def get_callback(config_path, X_train):
    unique_name = get_unique_name('tesorBoard_logs')
    config = read_config(config_path)
    artifacts = config['artifacts']
    artifacts_dir = artifacts['artifact_dir']
    ckpd_dir = artifacts['ckpd_dir']
    logs = config['logs']
    logs_dr = logs['logs_dir']
    TENSORBOARD_ROOT_LOG_DIR = logs['TENSORBOARD_ROOT_LOG_DIR']
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(artifacts_dir, logs_dr, TENSORBOARD_ROOT_LOG_DIR, unique_name) #for dir like artifacts/logs/TENSORBOARD_ROOT_LOG_DIR/unique_name
    os.makedirs(TENSORBOARD_ROOT_LOG_DIR, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_ROOT_LOG_DIR)
    file_writer = tf.summary.create_file_writer(logdir=TENSORBOARD_ROOT_LOG_DIR)
    with file_writer.as_default():
        images = np.reshape(X_train[20:40], (-1, 28, 28, 1))
        tf.summary.image('20 handwritten digit images', images, step=0)
    params = config['params']
    earlyStopping_callback = tf.keras.callbacks.EarlyStopping(patience=params['patience'],restore_best_weights=params['restore_best_weights'])
    ckpd_model_name = artifacts['ckpd_model_name']
    ckpd_model_dir = os.path.join(artifacts_dir, ckpd_dir)
    os.makedirs(ckpd_model_dir, exist_ok=True)
    # ckpd_path = os.path.join(ckpd_model_dir, ckpd_model_name)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpd_model_dir+"/"+ckpd_model_name, save_best_only=True)

    return[tensorboard_callback, earlyStopping_callback, ckpt_callback]
