import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
import numpy as np
import cv2


class PilotNet():
    def __init__(self, height: int, width: int, model_file: str='') -> None:
        self.image_height = height
        self.image_width = width
        
        if model_file:
            self.model = self.load_model(model_file)
            self.model.summary()
            print("model_loaded")
        else:
            self.model = self.build_model()

    def forward(self, data: np.ndarray, b_s: int=1):
        return self.model.predict(data, batch_size=b_s)

    def build_model(self):

        model = Sequential([
            layers.Rescaling(1./255, input_shape=(66, 200, 3)),
            layers.Conv2D(24, 5, strides=(2,2), activation="relu"),
            layers.Conv2D(36, 5, strides=(2,2), activation="relu"),
            layers.Conv2D(48, 5, strides=(2,2), activation="relu"),
            layers.Conv2D(64, 3, strides=(1,1), activation="relu"),
            layers.Conv2D(64, 3, strides=(1,1), activation="relu"),
            layers.Flatten(),
            layers.Dense(100, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(50, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(10, activation="relu"),
            layers.Dense(1, activation="relu")
        ])

        lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=10000, decay_rate=0.9)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_decay),
                      loss="mse",
                      metrics="mae")
        model.summary()
        return model
    
    def load_model(self, model_file: str):
        try:
            model = tf.keras.models.load_model(model_file)
        except IOError:
            print("Failed to load the model")
            SystemExit()
        return model
    def save_model(self, ck_type: str=''):
        if ck_type == 'h5':
            self.model.save('pilotnet_model.tf', save_format='h5')
        elif ck_type == 'tf':
            self.model.save('pilotnet_model', save_format='tf')
        else:
            self.model.save('pilotnet_model.keras')
