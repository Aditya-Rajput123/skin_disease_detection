import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from PIL import Image

# Load model
model = tf.keras.models.load_model('model/trained_model_98.h5', compile=False)

CLASS_NAMES = [
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

print(f"{'Filename':50} | {'Prediction':35} | {'Conf':7}")
print("-" * 100)

for f in files[:10]:
    full_path = os.path.join(uploads_dir, f)
    try:
        img = image.load_img(full_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x) # VGG preprocessing (Mean subtraction, BGR)
        
        preds = model.predict(x, verbose=0)[0]
        idx = np.argmax(preds)
        print(f"{f:50} | {CLASS_NAMES[idx]:35} | {preds[idx]*100:6.2f}%")
    except:
        continue
