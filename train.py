from model import linear_model, train_features, train_labels, test_features, test_labels, build_deep_model, build_medium_model
import tensorflow as tf
import pandas as pd
from model import shallow_medium_model, build_medium_model_sigmoid

# Linear Models
# model = linear_model()
# history = model.fit(train_features, train_labels, validation_data=(test_features, test_labels), epochs=100, verbose=2)

# Shallow Medium Model
# model = shallow_medium_model()

# Deep Model
SAVED_MODEL_PATH = '/Users/royakash/Documents/GitHub/MRR-SodaLimeSilicaGlass/trained_models/model_medium_4-12-5-1.h5'
# model = build_deep_model()

# Medium Model RELU + Adam as an optimizer
model = build_medium_model(1)

# Callbacks to save only the best model
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, verbose=2, mode='max')
save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    SAVED_MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(train_features, train_labels, validation_data=(
    test_features, test_labels), epochs=15, verbose=1, callbacks=[lr_reduce, save_checkpoint])

# Prediction
# model = tf.keras.models.load_model(SAVED_MODEL_PATH)
test_predictions = model.predict(test_features)

print(test_predictions)
print('################################')
print(test_labels)
print('################################')
print(model.summary())
