import json
import os
import tensorflow as tf

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

print("Model output shape:", model.output_shape)
print("Number of defined classes:", len(CLASS_NAMES))

print("Looking for training class indices...")

# Check if model has class indices
if hasattr(model, 'class_indices'):
    print(model.class_indices)
elif os.path.exists('model/class_indices.json'):
    print("Found model/class_indices.json.")
    with open('model/class_indices.json', 'r') as f:
        print(json.load(f))
else:
    print("No class_indices metadata found. Generating one based on CLASS_NAMES...")
    # Generate class_indices.json
    class_indices = {name: i for i, name in enumerate(CLASS_NAMES)}
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    with open('model/class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=4)
    print("Created model/class_indices.json successfully.")
    print("Class indices:")
    print(class_indices)
