from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model
import os

def training(config_path):
    config = read_config(config_path)
    validation_data_size = config['params']["validation_data_size"]
    (X_train, y_train), (X_valid, y_yalid), (X_test, y_test) = get_data(validation_data_size)
    validation_set = (X_valid, y_yalid)
    OPTIMIZER = config['params']["optimizer"]
    LOSS_FUNCTION = config['params']["loss_function"]
    num_classes = config['params']["num_classes"]
    metrics = config['params']["metrics"]
    epochs = config['params']['epochs']
    model_cf = create_model(OPTIMIZER, LOSS_FUNCTION, num_classes,metrics)
    history = model_cf.fit(X_train, y_train, epochs=epochs, validation_data=validation_set)
    # acc =  model_cf.evaluate(X_test, y_test)
    # print(acc)
    model_name = config['artifacts']['model_name']
    artifact_dir = config['artifacts']['artifact_dir']
    model_dir = config['artifacts']['model_dir']
    model_path = os.path.join(artifact_dir, model_dir)
    os.makedirs(model_path, exist_ok =True)
    save_model(model_cf,model_name,model_path)

