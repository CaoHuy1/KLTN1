import cv2
import os
from ultralytics import YOLO
import numpy as np
import preprocess_image as pre

# Load YOLO model
model = YOLO("best.pt")

# Load pre-trained SVM model for character recognition
svm_model = cv2.ml.SVM_load('svm.xml')

# Flags for cropped plate tracking and recognition state
license_plate_cropped = False
recognition_active = True

# Directory to save results
save_dir = r"D:\KLTN\results"
os.makedirs(save_dir, exist_ok=True)

# Define area for license plate detection
rect_x1, rect_y1, rect_x2, rect_y2 = 250, 250, 400, 350

# Define allowed characters for license plates
char_list = '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Configure parameters for SVM model
digit_w, digit_h = 32,32  # Character dimensions


def fine_tune(lp):
    return ''.join([char for char in lp if char in char_list])


def preprocess_svm_image(cropped_image):
    # Convert the cropped license plate image to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('blurred', blurred)

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(blurred,  255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imshow('binary_image', binary_image)

    img_binary_lp = cv2.erode(binary_image, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    # Apply morphological operations to close small gaps
    #kernel = np.ones((3, 3), np.uint8)
    #processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('processed_image', img_binary_lp)

    return img_binary_lp

def extract_text_with_svm(cropped_image):
    #processed_image = preprocess_svm_image(cropped_image)
    #cv2.imshow('processed_image', processed_image)

    #processed_image1 = pre.preprocess_image(cropped_image)
    #cv2.imshow('processed_image1', processed_image1)

    processed_image2 = pre.preprocess_image_2(cropped_image)
    cv2.imshow('processed_image2', processed_image2)

    #processed_image3 = pre.preprocess_image_1(cropped_image)
    #cv2.imshow('processed_image3', processed_image3)

    contours, _ = cv2.findContours(processed_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plate_info = ""
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Filter for valid character contours
        if w > 5 and h > 10:
            char_image = processed_image2[y:y + h, x:x + w]
            char_image_resized = cv2.resize(char_image, (digit_w, digit_h))  # Resize to the correct dimensions
            char_image_flattened = char_image_resized.reshape(1, -1).astype(np.float32)  # Flatten and convert to float32

            # Print the shape of the flattened image for debugging
            print(f"Shape of flattened image: {char_image_flattened.shape}")

            # Predict character using SVM
            result = svm_model.predict(char_image_flattened)[1]
            prediction = int(result[0, 0])

            # Convert prediction to ASCII if needed
            if prediction <= 9:  # If the result is a digit
                predicted_char = str(prediction)
            else:  # If it's an ASCII character
                predicted_char = chr(prediction + 55)  # Adjust offset for ASCII

            plate_info += predicted_char
            print("plate_info:", predicted_char)

    return fine_tune(plate_info)


def custom_predict():
    global license_plate_cropped, recognition_active

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame from camera")
            break

        # Draw the rectangular area
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)

        # Crop the frame to the defined area
        cropped_frame = frame[rect_y1:rect_y2, rect_x1:rect_x2]

        if recognition_active:
            results = model.predict(source=cropped_frame, conf=0.5, classes=[0], save=True, save_crop=True,
                                    project=save_dir, name="detect", exist_ok=True)

            if not license_plate_cropped:
                cropped_images_dir = os.path.join(save_dir, "detect", "crops", "numplate")

                if os.path.exists(cropped_images_dir) and os.listdir(cropped_images_dir):
                    cropped_image_path = \
                        sorted([os.path.join(cropped_images_dir, f) for f in os.listdir(cropped_images_dir)],
                               key=os.path.getmtime)[-1]
                    print(f"Cropped license plate saved at: {cropped_image_path}")

                    cropped_image = cv2.imread(cropped_image_path)
                    if cropped_image is not None:
                        plate_info = extract_text_with_svm(cropped_image)
                        print(f"Extracted plate info using SVM: {plate_info}")
                        license_plate_cropped = True
                        recognition_active = False
                    else:
                        print("Failed to load the cropped image.")
                else:
                    print("No cropped image found.")

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recognition_active = True
            license_plate_cropped = False
            print("Resuming recognition.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run prediction function
custom_predict()
