import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

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
if os.path.exists(uploads_dir):
    files = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
else:
    files = []

for idx, f in enumerate(files[:5]):
    img = image.load_img(f, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    print(f"\n--- Image {os.path.basename(f)} ---")
    
    preds_raw = model.predict(img_array, verbose=0)
    print(f"RAW: {CLASS_NAMES[np.argmax(preds_raw[0])]} ({np.max(preds_raw[0])*100:.2f}%)")

    preds_norm = model.predict(img_array / 255.0, verbose=0)
    print(f"/255: {CLASS_NAMES[np.argmax(preds_norm[0])]} ({np.max(preds_norm[0])*100:.2f}%)")

print("Done testing.")
