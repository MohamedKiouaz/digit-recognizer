import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers, models, optimizers

#%%

train_data = np.genfromtxt('train.csv', delimiter=',', skip_header = True)
test_data = np.genfromtxt('test.csv', delimiter=',', skip_header = True)

X = np.copy(train_data[:, 1:])
Y = np.copy(train_data[:, 0])

X = X.reshape(-1, 28, 28, 1)
Y = tf.keras.utils.to_categorical(Y)

X_test = np.copy(test_data).reshape(-1, 28, 28, 1)

#%%

input_layer = layers.Input(shape = (28, 28, 1, ))
x = layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', kernel_initializer = initializers.glorot_uniform())(input_layer)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x_shortcut_layer = x

for i in range(30):
    x = x_shortcut_layer
    x_shortcut_layer = layers.Conv2D(filters = 1, kernel_size = (1, 1), padding = 'same', kernel_initializer = initializers.glorot_uniform())(x_shortcut_layer)
    
    x = layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', kernel_initializer = initializers.glorot_uniform())(x)
    shortcut = x
    x = layers.BatchNormalization()(x)
    x_shortcut_layer = layers.Add()([x_shortcut_layer, x])
    
    x = layers.Activation('relu')(x_shortcut_layer)
    
x = layers.Flatten()(x)

output_layer = layers.Dense(Y.shape[1], activation = 'softmax')(x)

model = models.Model(inputs = input_layer, outputs = output_layer)
model.compile(optimizer = optimizers.Adam(), loss = 'categorical_crossentropy', metrics = ['acc'])
model.summary()

#%%

tf.keras.utils.plot_model(model, show_shapes=True)

#%%

history = model.fit(X, Y, epochs = 30, verbose = 2, validation_split = .1, batch_size = 2000)

#%%

Y_test = model.predict(X_test)

Y_test = np.argmax(Y_test, -1)
predictions = np.zeros((Y_test.size, 2), dtype = int)
predictions[:, 0] = np.arange(1, Y_test.size + 1)
predictions[:, 1] = Y_test

np.savetxt("prediction.csv", predictions, delimiter = ',', header = "ImageId,Label", fmt = "%i", comments = "")
