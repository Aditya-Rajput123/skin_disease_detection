import sys
import tensorflow as tf

with open('summary.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    print("Extracting model layers...")
    model = tf.keras.models.load_model('model/trained_model_98.h5', compile=False)
    model.summary()

    # Check if there is a rescaling layer internally
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
        config = layer.get_config()
        print("  Config:", config)
