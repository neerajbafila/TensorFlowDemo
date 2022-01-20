import tensorflow as tf


def get_data(validation_data_size):
    """This method is used to get MIST dataset
        and return MIST dataset
    """
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # create validation data set("validation_data_size from config file" data points)
    # as image is 8 bit, Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
    X_valid, X_train = X_train_full[:validation_data_size] / 255, X_train_full[validation_data_size:] / 255
    y_yalid, y_train = y_train_full[:validation_data_size], y_train_full[validation_data_size:]

    # scale the test data also
    X_test = X_test / 255

    return (X_train, y_train), (X_valid, y_yalid), (X_test, y_test)


# (X_train, y_train), (X_valid, y_yalid), (X_test, y_test) = get_data(5000)

# print(X_train)