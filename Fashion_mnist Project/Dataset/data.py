import pandas as pd 
import tensorflow as tf
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])

X_flat = X.reshape(X.shape[0], -1)

df = pd.DataFrame(X_flat)
df['label'] = y


df.to_csv('fashion_mnist.csv', index=False)