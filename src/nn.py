import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing
import keras

from keras.models import Sequential
from keras.layers.core import Dense, MaxoutDense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2

from sknn.mlp import Classifier, Layer

def run_nn(quality='low'):
    """
    Runs a simple neural network model; first fits the model
    on the training data (70 percent of the total data) and tests on 
    the rest of the data.

    Args:
        none

    Returns:
        none
    """

    data = io.load_data(quality=quality)
    X, y, class_names = preprocessing.create_data_tensor(data)  
    X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(X, y, test_size=0.3, shuffle=False)

    y_train_one_hot = np.zeros((y_train.shape[0], len(class_names)))
    for i in range(y_train.shape[0]):
        y_train_one_hot[i, y_train[i]] = 1

    y_test_one_hot = np.zeros((y_test.shape[0], len(class_names)))
    for i in range(y_test.shape[0]):
        y_test_one_hot[i, y_test[i]] = 1

    # flatten data
    flattened_Xtrain = preprocessing.flatten_matrix(X_train) 
    flattened_Xtest = preprocessing.flatten_matrix(X_test)

    # fit neural network model
    HIDDEN_LAYER_SIZE = y_train_one_hot.shape[1]
    # HIDDEN_LAYER_SIZE = y_train_one_hot.shape[1]*2

    ############ FAILED ##################
    # HIDDEN_LAYER_SIZE = X_train.shape[1]
    # 
    ######################################
    nn_model = Sequential()

    ####################################### MODELS #######################################
    
    ####################################### MODEL1 #######################################
    # Training Error: 0.517945411562
    # Testing Error: 0.778446115288
    # Epochs: 200
    # 
    nn_model.add(Dense(HIDDEN_LAYER_SIZE, input_dim=flattened_Xtrain.shape[1], init='uniform', activation="tanh"))
    nn_model.add(Dropout(0.2))
    nn_model.add(Dense(y_train_one_hot.shape[1], init='uniform', activation="tanh"))
    N_EPOCHS = 200
    ####################################### MODEL1 #######################################

    ####################################### MODEL2 #######################################
    # Training Error: 0.
    # Testing Error: 0.
    # 
    # nn_model.add(Dense(HIDDEN_LAYER_SIZE, input_dim=flattened_Xtrain.shape[1], init='uniform', activation="tanh"))
    # nn_model.add(Dropout(0.2))
    # nn_model.add(Dense(HIDDEN_LAYER_SIZE*10, activation="tanh", init='uniform'))
    # nn_model.add(Dropout(0.3))
    # nn_model.add(Dense(HIDDEN_LAYER_SIZE*2, activation="tanh", init='uniform'))
    # nn_model.add(Dense(y_train_one_hot.shape[1], init='uniform', activation="tanh"))
    ####################################### MODEL2 #######################################

    
    

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    nn_model.compile(loss='mean_squared_error', optimizer=sgd)

    nn_model.fit(flattened_Xtrain, y_train_one_hot, nb_epoch=N_EPOCHS)
    y_predict_train = nn_model.predict_classes(flattened_Xtrain)
    y_predict = nn_model.predict_classes(flattened_Xtest)
    # y_predict_one_hot = nn_model.predict(flattened_Xtest)

    # print metrics and confusion plot
    analysis.run_analyses(y_predict_train, y_train, y_predict, y_test, class_names)

if __name__ == '__main__':
    run_nn('low')
