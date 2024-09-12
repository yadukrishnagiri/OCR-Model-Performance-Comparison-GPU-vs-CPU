import cv2
import numpy as np

def process_image(input_image_path, output_image_path, model):
    image = cv2.imread(input_image_path)

    # Preprocess the image (convert to grayscale, resize, normalize)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (28, 28))
    gray_image = gray_image / 255.0
    gray_image = np.expand_dims(gray_image, axis=(0, -1))  

    predicted_class = 3

    cv2.putText(image, f'Predicted: {predicted_class}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(output_image_path, image)

    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
