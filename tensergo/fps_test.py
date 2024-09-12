import time
import cv2
from ocr_model import load_ocr_model
from image_processor import process_image

def calculate_fps(model, image_path, iterations=10):
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        process_image(image_path, 'output_image.jpg', model)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    # Calculate average FPS
    avg_time_per_frame = total_time / iterations
    fps = 1 / avg_time_per_frame
    return fps

def main():
    
    model = load_ocr_model('ocr_model.h5')

    image_path = 'input_image.jpg'

    fps = calculate_fps(model, image_path)
    print(f'Average FPS: {fps:.2f} frames per second')

if __name__ == '__main__':
    main()
