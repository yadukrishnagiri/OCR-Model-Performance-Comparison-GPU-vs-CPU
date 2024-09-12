import tensorflow as tf
from tensorflow.keras.models import load_model

def load_ocr_model(model_path):
    """
    Load the OCR model. This is where you ensure the model runs on the CPU.
    """
    with tf.device('/CPU:0'):
        model = load_model(model_path)
    return model
