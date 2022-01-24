from gc import callbacks
import shutil
from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model
from src.utils.callback import get_callback
import os

def training(config_path):
    config = read_config(config_path)
    validation_data_size = config['params']["validation_data_size"]
    (X_train, y_train), (X_valid, y_yalid), (X_test, y_test) = get_data(validation_data_size)
    validation_set = [X_valid, y_yalid]
    OPTIMIZER = config['params']["optimizer"]
    LOSS_FUNCTION = config['params']["loss_function"]
    num_classes = config['params']["num_classes"]
    metrics = config['params']["metrics"]
    epochs = config['params']['epochs']
    model_cf = create_model(OPTIMIZER, LOSS_FUNCTION, num_classes,metrics)
    CALL_BACK_LIST = get_callback(config_path, X_train)
    try:
        history = model_cf.fit(X_train, y_train, epochs=epochs, validation_data=validation_set, callbacks=CALL_BACK_LIST)
        print("training completed")
    except Exception as e:
        print('Exception occured ', e)

    # acc =  model_cf.evaluate(X_test, y_test)
    # print(acc)
    model_name = config['artifacts']['model_name']
    artifact_dir = config['artifacts']['artifact_dir']
    model_dir = config['artifacts']['model_dir']
    model_path = os.path.join(artifact_dir, model_dir)
    os.makedirs(model_path, exist_ok =True)
    try:
        lst_of_file = os.listdir(model_path)
        if  lst_of_file:
            os.makedirs(model_path+'Old', exist_ok=True)
            for i in lst_of_file:
                shutil.move(model_path+"/" +i, model_path+"old")
        try:
            save_model(model_cf,model_name,model_path)
            print(f'Model saved at {model_path}')
        except Exception as e:
            print('Exception occured while saving model', e)
    except Exception as e:
        print(e)
    

