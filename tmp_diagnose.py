import os
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model('model/trained_model_98.h5', compile=False)
uploads = 'static/uploads'
files = [f for f in os.listdir(uploads) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"{'Filename':50} | {'IDX':3} | {'CONF':7}")
print("-" * 65)

for f in files:
    try:
        path = os.path.join(uploads, f)
        # Testing RAW RGB
        img = Image.open(path).convert('RGB').resize((128, 128))
        x = np.array(img).astype('float32')
        preds = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
        idx = np.argmax(preds)
        print(f"{f:50} | {idx:3} | {preds[idx]*100:6.2f}%")
    except Exception as e:
        # print(f"Error {f}: {e}")
        pass
