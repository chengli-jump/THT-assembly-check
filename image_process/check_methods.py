import cv2


#色调、饱和度阈值筛选
def check_by_gray(box_slice_image, box_slice_image_golden):
    hsv_image = get_hsv_from_box(box_slice_image)
    num_pix = len(hsv_image)
    valid_pix_count = 0
    gray_hue_threshold = 90
    gray_sat_threshold = 80
    for pix in hsv_image:
        h, s, v = pix
        if h > gray_hue_threshold and s < gray_sat_threshold:
            valid_pix_count += 1
    white_total = num_pix - valid_pix_count
    white_percent = round((num_pix - valid_pix_count) / num_pix * 100 , 2)
    return white_total, white_percent
#亮度筛选
def check_by_black(box_slice_image, box_slice_image_golden):
    hsv_image = get_hsv_from_box(box_slice_image)
    num_pix = len(hsv_image)
    valid_pix_count = 0
    black_val_threshold = 70
    for pix in hsv_image:
        h, s, v = pix
        if v < black_val_threshold:
            valid_pix_count += 1
    white_total = num_pix - valid_pix_count
    white_percent = round((num_pix - valid_pix_count) / num_pix * 100 , 2)
    return white_total, white_percent

def check_by_gray_or_blue(box_slice_image, box_slice_image_golden):
    check_gray_result = check_by_gray(box_slice_image, box_slice_image_golden)
    check_blue_result = check_by_blue(box_slice_image, box_slice_image_golden)
    # return the result for the color that has smaller white percent. (Which means more similarity.)
    return check_gray_result if check_gray_result[1] < check_blue_result[1] else check_blue_result

def check_by_blue(box_slice_image, box_slice_image_golden):
    hsv_image = get_hsv_from_box(box_slice_image)
    num_pix = len(hsv_image)
    valid_pix_count = 0
    blue_hue_threshold = 80
    blue_sat_threshold = 180
    for pix in hsv_image:
        h, s, v = pix
        if h > blue_hue_threshold and s > blue_sat_threshold:
            valid_pix_count += 1
    white_total = num_pix - valid_pix_count
    white_percent = round((num_pix - valid_pix_count) / num_pix * 100 , 2)
    return white_total, white_percent

def check_by_red(box_slice_image, box_slice_image_golden):
    hsv_image = get_hsv_from_box(box_slice_image)
    num_pix = len(hsv_image)
    valid_pix_count = 0
    red_hue_threshold_1 = 30
    red_hue_threshold_2 = 150
    for pix in hsv_image:
        h, s, v = pix
        if h < red_hue_threshold_1 or h > red_hue_threshold_2:
            valid_pix_count += 1
    white_total = num_pix - valid_pix_count
    white_percent = round((num_pix - valid_pix_count) / num_pix * 100 , 2)
    return white_total, white_percent

def check_find_hat_center(box_slice_image, box_slice_image_golden):
    cv2.imwrite("hat_slice.png", box_slice_image)
    box_slice_image_gray = cv2.cvtColor(box_slice_image, cv2.COLOR_RGB2GRAY)
    _, hat_gray = cv2.threshold(box_slice_image_gray, 50, 255, cv2.THRESH_BINARY)
    center_area = hat_gray[:, hat_gray.shape[1] // 5 * 2 : hat_gray.shape[1] // 5 * 3]
    white_total = len([px for px in center_area.flatten() if px == 255])
    total = len(center_area.flatten())
    white_percent = round(white_total / total * 100 , 2)
    return white_total, white_percent

def check_by_vertical_angle(box_slice_image, box_slice_image_golden):
    import numpy as np
    image_gray = cv2.cvtColor(box_slice_image, cv2.COLOR_RGB2GRAY)
    _, image_gray = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)
    kernel_for_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close_gray = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel_for_closing, iterations=1)
    upper = [close_gray.shape[0], 0]
    lower = [0, 0]
    for i in range(close_gray.shape[0]):
        for j in range(close_gray.shape[1]):
            if close_gray[i][j] == 0:
                if i <= upper[0]:
                    upper = [i, j]
                elif i >= lower[0]:
                    lower = [i, j]
                # upper = [min(upper[0], i), max(upper[1], j)]
                # lower = [max(lower[0], i), max(lower[1], j)]
    angle = np.arctan( abs(upper[1] - lower[1]) / abs(upper[0] - lower[0]) )
    white_percent = round(float(angle) * 100 , 2)
    return 0, white_percent

def get_hsv_from_box(box_slice_image):
    hsv_image = cv2.cvtColor(box_slice_image, cv2.COLOR_BGR2HSV)
    hsv_image = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))
    return hsv_image

# opencv_processing calls this function to get check results
def check_by_method(box_slice_image, box_slice_image_golden, threshold_method):
    white_total, white_percent = threshold_method_functions[threshold_method](box_slice_image, box_slice_image_golden)
    return white_total, white_percent

def test():
    pass


# Map check methods to check functions
threshold_method_functions = {
    "gray_count":  check_by_gray,
    "blue_count":  check_by_blue,
    "gray_or_blue":  check_by_gray_or_blue,
    "black_count": check_by_black,
    "red_count":   check_by_red,
    "hat_center":   check_find_hat_center,
    "vertical_angle": check_by_vertical_angle,
    "test":test,
}
