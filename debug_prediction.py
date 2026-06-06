import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess, decode_predictions
import numpy as np
import os
import cv2

# Load models
model = tf.keras.models.load_model('model/trained_model_98.h5', compile=False)
gatekeeper = MobileNetV2(weights='imagenet')

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
test_images = [
    '2_20260311164123_07RosaceaMilia0120.jpg',
    '2_20260311164911_07VascularFace0120.jpg',
    '2_20260312004742_hemangioma-22.jpg',
    '2_20260312005151_tinea-scalp-55.jpg'
]

def analyze(img_path):
    full_path = os.path.join(uploads_dir, img_path)
    if not os.path.exists(full_path):
        print(f"Skipping {img_path}, not found.")
        return

    print(f"\n{'='*20} ANALYZING: {img_path} {'='*20}")

    # --- GATEKEEPER CHECK ---
    img_gk = image.load_img(full_path, target_size=(224, 224))
    x_gk = image.img_to_array(img_gk)
    x_gk = np.expand_dims(x_gk, axis=0)
    x_gk = mobilenet_preprocess(x_gk)
    gk_preds = gatekeeper.predict(x_gk, verbose=0)
    decoded = decode_predictions(gk_preds, top=3)[0]
    print("Gatekeeper Top 3:")
    for i, res in enumerate(decoded):
        print(f"  {i+1}. {res[1]}: {res[2]*100:.2f}%")

    # --- SKIN MODEL CHECKS ---
    img_skin = image.load_img(full_path, target_size=(128, 128))
    x_rgb = image.img_to_array(img_skin)
    
    # BGR conversion
    x_bgr = cv2.cvtColor(x_rgb, cv2.COLOR_RGB2BGR)

    variations = {
        "RGB RAW (0-255)": x_rgb,
        "RGB Norm (0-1)": x_rgb / 255.0,
        "BGR RAW (0-255)": x_bgr,
        "BGR Norm (0-1)": x_bgr / 255.0
    }

    for name, data in variations.items():
        inp = np.expand_dims(data, axis=0)
        preds = model.predict(inp, verbose=0)[0]
        top_idx = np.argsort(preds)[-3:][::-1]
        print(f"\n{name}:")
        for idx in top_idx:
            print(f"  {CLASS_NAMES[idx]}: {preds[idx]*100:.2f}%")

for img in test_images:
    analyze(img)
