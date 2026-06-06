import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model/trained_model_98.h5', compile=False)

# Test 1: Zeros
out_zeros = model.predict(np.zeros((1, 128, 128, 3)))
print("Zeros max class:", np.argmax(out_zeros), "prob:", np.max(out_zeros))

# Test 2: Ones
out_ones = model.predict(np.ones((1, 128, 128, 3)))
print("Ones max class:", np.argmax(out_ones), "prob:", np.max(out_ones))

# Test 3: Random
np.random.seed(42)
out_rand = model.predict(np.random.rand(1, 128, 128, 3))
print("Random max class:", np.argmax(out_rand), "prob:", np.max(out_rand))

# Test 4: Random * 255 (no division by 255)
out_rand255 = model.predict(np.random.rand(1, 128, 128, 3) * 255.0)
print("Rand255 max class:", np.argmax(out_rand255), "prob:", np.max(out_rand255))
