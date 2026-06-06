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
files = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
files.sort(key=os.path.getmtime, reverse=True)

if not files:
    print("No images to test.")
    exit()

img_path = files[0]
print("Testing on:", img_path)

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# 1. No scaling
preds_raw = model.predict(img_array)
idx_raw = np.argmax(preds_raw[0])
print(f"RAW (No scale) => {CLASS_NAMES[idx_raw]} ({preds_raw[0][idx_raw]*100:.2f}%)")

# 2. / 255.0
preds_255 = model.predict(img_array / 255.0)
idx_255 = np.argmax(preds_255[0])
print(f"/ 255.0 => {CLASS_NAMES[idx_255]} ({preds_255[0][idx_255]*100:.2f}%)")

# 3. [-1, 1]
preds_1 = model.predict((img_array / 127.5) - 1)
idx_1 = np.argmax(preds_1[0])
print(f"[-1, 1] => {CLASS_NAMES[idx_1]} ({preds_1[0][idx_1]*100:.2f}%)")

# 4. VGG preprocess
try:
    from tensorflow.keras.applications.vgg16 import preprocess_input
    preds_vgg = model.predict(preprocess_input(np.copy(img_array)))
    idx_vgg = np.argmax(preds_vgg[0])
    print(f"VGG preprocess => {CLASS_NAMES[idx_vgg]} ({preds_vgg[0][idx_vgg]*100:.2f}%)")
except Exception as e:
    print("VGG failed:", e)

with open('prep_result.txt', 'w') as f:
    f.write(f"RAW: {CLASS_NAMES[idx_raw]}\n")
    f.write(f"255: {CLASS_NAMES[idx_255]}\n")
    f.write(f"1: {CLASS_NAMES[idx_1]}\n")
