import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers, models, optimizers
import matplotlib.pyplot as plt

#%%

train_data = pd.read_csv('train.csv', delimiter=',').to_numpy()
test_data = pd.read_csv('test.csv', delimiter=',').to_numpy()

X = np.copy(train_data[:, 1:]) / 255.0
Y = np.copy(train_data[:, 0]) / 255.0

X = X.reshape(-1, 28, 28, 1)
Y = tf.keras.utils.to_categorical(Y)

X_test = np.copy(test_data).reshape(-1, 28, 28, 1) / 255.0

#%%

input_layer = layers.Input(shape = (28, 28, 1, ))

N = 2

x = input_layer
for i in range(N):
    x = layers.Conv2D(filters = max(1, N-i), kernel_size = (3, 3), padding = 'same', kernel_initializer = initializers.glorot_uniform())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

encoder_output = x

for i in range(N):
    x = layers.Conv2DTranspose(filters = max(1, N-i), kernel_size = (3, 3), padding = 'same', kernel_initializer = initializers.glorot_uniform())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2DTranspose(filters = 1, kernel_size = (1, 1), padding = 'same', kernel_initializer = initializers.glorot_uniform())(x)
output_layer = layers.Activation('sigmoid')(x)

model = models.Model(inputs = input_layer, outputs = output_layer)
model.compile(optimizer = optimizers.Adam(), loss = 'mse', metrics = ['mae'])
model.summary()

#%%

tf.keras.utils.plot_model(model, show_shapes=True)

#%%
for j in range(100):
    history = model.fit(X, X, epochs = 1, verbose = 2, batch_size = 2000)

    Y_test = model.predict(X_test)
    plt.close('all') 
    for i in range(5):
        N = np.random.randint(0, 20000)
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(X_test[N, :, :, 0])
        ax[1].imshow(Y_test[N, :, :, 0])
        fig.savefig(str(N) + ".pdf")

