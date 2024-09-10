from model import PilotNet
from data import Data
import tensorboard
import datetime
import tensorflow as tf


nvdriver=PilotNet(height=66, width=200)
print("data preparing")
data = Data((200, 66))
print("data preparation complete")

model_name = "nvdriver"

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# Setting checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(f"./models/{model_name}", monitor="loss", verbose=1, save_best_only=True)

epochs = 30
steps_per_epoch = 10
steps_val = 10
batch_size = 64

x_train, y_train = data.get_training_data()
x_test, y_test = data.get_test_data()

print("start training")
nvdriver.model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=steps_val,
        validation_split=0.2,
        callbacks=[tensorboard_callback, checkpoint]

)
print("training over")