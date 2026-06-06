import tensorflow as tf
model = tf.keras.models.load_model('model/trained_model_98.h5', compile=False)
model.summary()
