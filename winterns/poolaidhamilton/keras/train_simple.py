from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv1D
from keras.models import model_from_json, model_from_yaml
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../')
from readercleaner import get_data

# rather, for pool
output_dim = 1
model_dir = 'simple_model.json'
weights_dir = 'simple_wts.h5'
train_data = get_data(0,100)
train_data = train_data.sample(frac=1).reset_index(drop=True)
X_train = train_data[['x','y','cuex','cuey']].as_matrix()
Y_train = train_data[['diff']].as_matrix()
test_data = get_data(100,120)
test_data = test_data.sample(frac=1).reset_index(drop=True)
X_test = test_data[['x','y','cuex','cuey']].as_matrix()
Y_test = test_data[['diff']].as_matrix()


# is it ok if i don't convert class vectors to binary class matrices?
# Y_train = np_utils.to_categorical(Y_train, 3)
# Y_test = np_utils.to_categorical(Y_test, 3)

print("X size", X_train.shape)
print("Y size", Y_train.shape)

# TODO: Duncan pls Change the model to your heart's delight!
# And if you would like to scale the difficulties up, see my comment in the diff1 function in readercleaner
model = Sequential()
#model.add(Conv1D(1, 2, strides=2, input_shape = (32,1), padding = 'same', activation = 'sigmoid'))
#model.add(Flatten())
#model.add(Dense(10,input_dim = 4, activation='sigmoid'))
#model.add(Dense(10, activation='sigmoid'))
#model.add(Dense(10, activation='sigmoid'))
#model.add(Dense(5, activation='sigmoid'))
model.add(Dense(25, input_dim = 4, activation='sigmoid'))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(output_dim, activation='sigmoid'))
model.summary()
batch_size = 200
nb_epoch = 40

# compile the model

sgd = optimizers.SGD() # added a higher learning rate
model.compile(optimizer=sgd, loss='mean_squared_error') # changed optim, error
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score)

# save losses
loss_history = history.history["loss"]
np_loss_history = np.array(loss_history)
print(np_loss_history.shape)
np.save('simple_history',np_loss_history)

# save model and weights

json_string = model.to_json() # as json
open(model_dir, 'w').write(json_string)
# yaml_string = model.to_yaml() #as yaml
# open('mnist_Logistic_model.yaml', 'w').write(yaml_string)

# save the weights in h5 format
model.save_weights(weights_dir)

# uncomment the code below (and modify accordingly) to read a saved model and weights
model = model_from_json(open(model_dir).read())# if json
# model = model_from_yaml(open('mnist_Logistic_model.yaml').read())# if yaml
model.load_weights(weights_dir)


