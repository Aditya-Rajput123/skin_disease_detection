import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess, decode_predictions
import numpy as np
import os
import glob

# Load models
MODEL_PATH = 'model/trained_model_98.h5'
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
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

allowed_gk_terms = ['band_aid', 'mask', 'face_powder', 'lipstick', 'wool', 'velvet', 'swab']

uploads_dir = 'static/uploads'
list_of_files = glob.glob(os.path.join(uploads_dir, '*'))
# Sort by modification time
# list_of_files.sort(key=os.path.getmtime)
recent_images = [
    'static/uploads/2_20260311164232_acne-cystic-61.jpg',
    'static/uploads/2_20260312005151_tinea-scalp-55.jpg',
    'static/uploads/2_20260313115400_scleroderma-27.jpg',
    'static/uploads/2_20260313130453_hemangioma-44.jpg'
]

def analyze(full_path):
    img_name = os.path.basename(full_path)
    print(f"\n{'='*20} ANALYZING: {img_name} {'='*20}")

    # --- GATEKEEPER CHECK ---
    try:
        img_gk = image.load_img(full_path, target_size=(224, 224))
        x_gk = image.img_to_array(img_gk)
        x_gk = np.expand_dims(x_gk, axis=0)
        x_gk = mobilenet_preprocess(x_gk)
        gk_preds = gatekeeper.predict(x_gk, verbose=0)
        decoded = decode_predictions(gk_preds, top=3)[0]
        
        top_gk_conf = decoded[0][2] * 100
        top_gk_label = decoded[0][1].lower()
        
        is_object = False
        if top_gk_conf > 30.0:
            if top_gk_label not in allowed_gk_terms:
                is_object = True
        
        print(f"Gatekeeper (Is Object: {is_object}):")
        for i, res in enumerate(decoded):
            print(f"  {i+1}. {res[1]}: {res[2]*100:.2f}%")
    except Exception as e:
        print(f"Gatekeeper Error: {e}")
        is_object = False

    # --- SKIN MODEL CHECKS ---
    import cv2
    try:
        img_skin = image.load_img(full_path, target_size=(128, 128))
        x_rgb = image.img_to_array(img_skin)
        x_bgr = cv2.cvtColor(x_rgb, cv2.COLOR_RGB2BGR)

        variations = {
            "RGB RAW (0-255)": x_rgb,
            "RGB Norm (0-1)": x_rgb / 255.0,
            "BGR RAW (0-255)": x_bgr,
            "BGR Norm (0-1)": x_bgr / 255.0
        }

        print("\nSkin Model Variations:")
        best_conf = 0
        best_variant = ""

        for name, data in variations.items():
            inp = np.expand_dims(data, axis=0)
            preds = model.predict(inp, verbose=0)[0]
            top_idx = np.argmax(preds)
            conf = preds[top_idx] * 100
            print(f"  {name}: {CLASS_NAMES[top_idx]} ({conf:.2f}%)")
            if conf > best_conf:
                best_conf = conf
                best_variant = name

        is_low_conf = best_conf < 50.0
        print(f"\nBest Variant: {best_variant} ({best_conf:.2f}%)")
            
    except Exception as e:
        print(f"Skin Model Error: {e}")

    # FINAL STATUS
    if is_object:
        print("\nRESULT: REJECTED by Gatekeeper")
    elif best_conf < 30.0: # New potential lower threshold
        print(f"\nRESULT: REJECTED by Confidence Threshold ({best_conf:.2f}% < 30%)")
    else:
        print(f"\nRESULT: ACCEPTED (Confidence: {best_conf:.2f}%)")

for img_path in recent_images:
    analyze(img_path)
