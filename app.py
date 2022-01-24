from src.training import training
from prediction import  predict
from src.utils.data_mgmt import get_data
from src.utils.common import read_config
if __name__=='__main__':
    config_file = 'config.yaml' 

    # training(config_file)
   
   
    # for testing 
    config = read_config(config_file)
    validation_data_size = config['params']["validation_data_size"]
    (X_train, y_train), (X_valid, y_yalid), (X_test, y_test) = get_data(validation_data_size)
    predict(X_test[:10], config_file)