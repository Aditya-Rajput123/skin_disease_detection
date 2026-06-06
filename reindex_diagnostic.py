import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Load model
model = tf.keras.models.load_model('model/trained_model_98.h5', compile=False)

CLASS_NAMES_GUESS = [
    'Acne and Rosacea Photos', 'Actinic Keratosis...', 'Atopic Dermatitis Photos',
    'Cellulitis Impetigo...', 'Eczema Photos', 'Exanthems and Drug Eruptions',
    'Herpes HPV...', 'Light Diseases...', 'Lupus...', 
    'Melanoma Skin Cancer Nevi and Moles', 'Poison Ivy Photos...', 
    'Psoriasis pictures Lichen Planus...', 'Seborrheic Keratoses...', 
    'Systemic Disease', 'Tinea Ringworm...', 'Urticaria Hives', 
    'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum...'
]

uploads_dir = 'static/uploads'
files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]

results = []

for f in files:
    full_path = os.path.join(uploads_dir, f)
    try:
        img = Image.open(full_path).convert('RGB').resize((128, 128))
        x = np.array(img).astype('float32') # RAW pixels
        inp = np.expand_dims(x, axis=0)
        preds = model.predict(inp, verbose=0)[0]
        top_idx = np.argmax(preds)
        conf = preds[top_idx]
        results.append((f, top_idx, conf))
    except:
        continue

# Sort by index
results.sort(key=lambda x: x[1])

print(f"{'Filename':50} | {'Index':5} | {'Conf':7}")
print("-" * 65)
for f, idx, conf in results:
    print(f"{f:50} | {idx:5} | {conf*100:6.2f}%")
