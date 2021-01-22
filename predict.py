from model import train_features, train_labels, test_features, test_labels
import tensorflow as tf

model = tf.keras.models.load_model('/Users/royakash/Documents/GitHub/MRR-SodaLimeSilicaGlass/trained_models/model_medium_ep3000.h5')
test_predictions = model.predict(test_features)

mrrs = [data[0] for data in test_predictions]
actual_mrrs = [data[0] for data in test_labels.to_numpy()]

from matplotlib import pyplot as plt
plt.plot(mrrs, actual_mrrs)
plt.show()