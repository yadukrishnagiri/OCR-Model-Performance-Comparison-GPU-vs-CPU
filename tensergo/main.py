# main.py
import time
import tensorflow as tf
from ocr_model import load_ocr_model
from image_processor import process_image
from tensorflow.keras import datasets

def test_accuracy(model):
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return test_accuracy

def test_fps(model, input_image_path, num_iterations=10):
    start_time = time.time()
    
    for _ in range(num_iterations):
        process_image(input_image_path, "output_image.jpg", model)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = num_iterations / total_time
    
    return avg_fps

def main():
    input_image = 'input_image.jpg'

    print("Testing on CPU...")
    with tf.device('/CPU:0'):
        cpu_model = load_ocr_model('ocr_model.h5')
        cpu_accuracy = test_accuracy(cpu_model)
        cpu_fps = test_fps(cpu_model, input_image)
    
    if tf.config.list_physical_devices('GPU'):
        print("Testing on GPU...")
        with tf.device('/GPU:0'):
            gpu_model = load_ocr_model('ocr_model.h5')
            gpu_accuracy = test_accuracy(gpu_model)
            gpu_fps = test_fps(gpu_model, input_image)
    else:
        print("No GPU found. Skipping GPU test.")
        gpu_accuracy, gpu_fps = None, None

    print("\n===== Comparison Results =====")
    print(f"CPU Accuracy: {cpu_accuracy * 100:.2f}%")
    print(f"CPU FPS: {cpu_fps:.2f} frames/second")

    if gpu_accuracy is not None and gpu_fps is not None:
        print(f"GPU Accuracy: {gpu_accuracy * 100:.2f}%")
        print(f"GPU FPS: {gpu_fps:.2f} frames/second")
    else:
        print("GPU comparison skipped as no GPU is available.")
    
if __name__ == '__main__':
    main()
