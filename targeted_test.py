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

uploads_dir = 'static/uploads'
test_cases = [
    ('2_20260312004205_lupus-chronic-cutaneous-4.jpg', 'Lupus...'),
    ('2_20260311173205_09EczemaStaph020217.jpg', 'Eczema Photos'),
    ('2_20260312004810_nevoxanthoendothelioma-5.jpg', 'Melanoma Skin Cancer Nevi and Moles'),
    ('2_20260312004944_seborrheic-dermatitis-132.jpg', 'Eczema Photos'), # Seborrheic Derm often grouped or near Eczema
    ('2_20260312004912_psoriasis-scalp-89.jpg', 'Psoriasis pictures Lichen Planus...'),
]

def analyze_targeted(img_name, expected):
    full_path = os.path.join(uploads_dir, img_name)
    if not os.path.exists(full_path): return
    
    img = Image.open(full_path).convert('RGB').resize((128, 128))
    x = np.array(img).astype('float32') # RAW pixels

    inp = np.expand_dims(x, axis=0)
    preds = model.predict(inp, verbose=0)[0]
    top_indices = np.argsort(preds)[-3:][::-1]
    
    print(f"\n--- {img_name} (Expected similarity to: {expected}) ---")
    for i in top_indices:
        print(f"  {CLASS_NAMES[i]:35} | {preds[i]*100:6.2f}%")

for name, exp in test_cases:
    analyze_targeted(name, exp)
