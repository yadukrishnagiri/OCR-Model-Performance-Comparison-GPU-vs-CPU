import tensorflow as tf
from tensorflow.keras import datasets
from ocr_model import load_ocr_model

def evaluate_model(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    return test_accuracy

def main():
    # Load the test data (using MNIST dataset as an example)
    (_, _), (x_test, y_test) = datasets.mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    model = load_ocr_model('ocr_model.h5')

    accuracy = evaluate_model(model, x_test, y_test)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
