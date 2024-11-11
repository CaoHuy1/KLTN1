import cv2
import numpy as np
from torchvision.transforms.v2.functional import resize_image
from skimage.filters import sobel, roberts, scharr, prewitt


def preprocess_image(image_path):
    # Read the image
    #image = cv2.imread(image_path)
    # Resize the image to double the size
    scale_percent = 200  # Increase size by 200%
    width = int(image_path.shape[1] * scale_percent / 100)
    height = int(image_path.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image_path, (width, height), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    enhanced_image = cv2.equalizeHist(gray)

    # Apply binary thresholding
    _, thresholded_image = cv2.threshold(enhanced_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((1, 1), np.uint8)
    #remove_noise = cv2.erode(thresholded_image, kernel, iterations=1)
    remove_noise = cv2.dilate(thresholded_image, kernel, iterations=1)
    cv2.imshow("remove_noise", remove_noise)

    # Denoise the image
    denoised_image = cv2.GaussianBlur(remove_noise, (1, 1), 0)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(denoised_image, -1, kernel)
    cv2.imshow("Sharpened", sharpened_image)
    return sharpened_image

def preprocess_image_2(image):
    # Read the image
    #image = cv2.imread(image_path)
    # Resize the image to double the size
    scale_percent = 300  # Increase size by 200%
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    image = clahe_app(resized_image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    enhanced_image = cv2.equalizeHist(gray)

    # Apply binary thresholding
    _, thresholded_image_1 = cv2.threshold(enhanced_image, 190, 255, cv2.THRESH_BINARY_INV)

    # Denoise the image
    denoised_image = cv2.GaussianBlur(thresholded_image_1, (3, 3), 0)

    remove_noise_image = remove_noise(denoised_image)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(remove_noise_image, -1, kernel)

    cv2.imshow("Sharpened_2", sharpened_image)
    return sharpened_image


def clahe_app(image):
    lab_frame = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(l)
    updated_lab = cv2.merge((clahe_img, a, b))
    CLAHE_img = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)
    cv2.imshow("CLAHE", CLAHE_img)
    return CLAHE_img

def adaptive_theshold(image):
    # Convert to grayscale if the image is not already grayscale
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    frame = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 113, 1)
    cv2.imshow("adaptive_threshold", frame)
    return frame

def remove_noise(image):
    kernel = np.ones((1, 1), np.uint8)
    #frame = cv2.erode(image, kernel, iterations=1)
    frame = cv2.dilate(image, kernel, iterations=1)
    return frame

def preprocess_image_1(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #gray = clahe.apply(gray)

    # Xác định tỷ lệ phóng to (ở đây là 2 lần)
    #scale_factor = 2.0

    # Tính toán kích thước mới của ảnh
    #new_width = int(image.shape[1] * scale_factor)
    #new_height = int(image.shape[0] * scale_factor)
    #new_size = (new_width, new_height)

    # Phóng to ảnh
    #resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    #image = cv2.imread(image)

    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(V)

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow("bfilter", bfilter)
    #gray = clahe_app(bfilter)


    # Increase contrast
    enhanced_image = cv2.equalizeHist(bfilter)
    cv2.imshow("enhanced_img", enhanced_image)

    _, binary = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("process_3", binary)
    return binary


