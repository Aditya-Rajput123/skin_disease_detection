import tensorflow as tf
from tensorflow.keras.preprocessing import image
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

def get_top_pred(img_array):
    inp = np.expand_dims(img_array, axis=0)
    preds = model.predict(inp, verbose=0)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], preds[idx]

def diagnostic(img_path):
    print(f"\n--- Testing: {os.path.basename(img_path)} ---")
    
    # Load image once as RGB
    img_rgb_pil = Image.open(img_path).convert('RGB').resize((128, 128))
    img_rgb = np.array(img_rgb_pil).astype('float32')
    
    # BGR version
    img_bgr = img_rgb[:, :, ::-1]

    tests = [
        ("RGB RAW", img_rgb),
        ("RGB / 255", img_rgb / 255.0),
        ("BGR RAW", img_bgr),
        ("BGR / 255", img_bgr / 255.0),
    ]

    for name, data in tests:
        label, conf = get_top_pred(data)
        print(f"{name:10}: {label:35} | {conf*100:6.2f}%")

uploads_dir = 'static/uploads'
# Select a few diverse-looking filenames if possible
files = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
files.sort(key=os.path.getmtime, reverse=True)

for f in files[:8]:
    diagnostic(f)
