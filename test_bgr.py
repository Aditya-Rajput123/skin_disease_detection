import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2

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

for f in files[:3]:
    # 1. RGB (Keras default)
    img_rgb = image.load_img(f, target_size=(128, 128))
    img_array_rgb = image.img_to_array(img_rgb)
    img_array_rgb = np.expand_dims(img_array_rgb, axis=0)

    # 2. BGR (OpenCV default)
    img_bgr = cv2.imread(f)
    img_bgr = cv2.resize(img_bgr, (128, 128))
    img_array_bgr = np.expand_dims(img_bgr.astype('float32'), axis=0)

    print(f"\n--- {os.path.basename(f)} ---")
    
    preds_rgb = model.predict(img_array_rgb, verbose=0)
    print(f"RGB RAW: {CLASS_NAMES[np.argmax(preds_rgb[0])]} ({np.max(preds_rgb[0])*100:.2f}%)")

    preds_bgr = model.predict(img_array_bgr, verbose=0)
    print(f"BGR RAW: {CLASS_NAMES[np.argmax(preds_bgr[0])]} ({np.max(preds_bgr[0])*100:.2f}%)")
