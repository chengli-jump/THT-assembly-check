import cv2
import numpy as np
 
def _find_pcb_area_of_image(image_raw, is_golden_sample=False, hue_range=(45, 150), saturation_range=(20, 150), value_range=(0, 255)):
        hsv_image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2HSV)
        # If config file defines hsv range, use specified range instead.
        config = {'pcb_area_hue_range': '45-95','pcb_area_saturation_range': '10-150','pcb_area_value_range': '0-255','saturation_value': '0',
        'minimum_size_of_pcb_area': '400000','golden_sample_path': 'golden_images/golden_sample_Picture2.png'}
        try:
            if config['pcb_area_hue_range']:
                hue_range = config['pcb_area_hue_range'].strip().split('-')
            if config['pcb_area_saturation_range']:
                saturation_range = config['pcb_area_saturation_range'].strip().split('-')
            if config['pcb_area_value_range']:
                value_range = config['pcb_area_value_range'].strip().split('-')
        except:
            print('Failed when loading hsv ranges in pcb config. Please check the format.')
        # define range of color in HSV
        lower_blue = np.array([int(hue_range[0]), int(saturation_range[0]), int(value_range[0])])
        upper_blue = np.array([int(hue_range[1]), int(saturation_range[1]), int(value_range[1])])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(image_raw, image_raw, mask=mask)
        hsv_thres = cv2.adaptiveThreshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 0)
        kernel_for_opening = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        kernel_for_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        open_gray = cv2.morphologyEx(hsv_thres, cv2.MORPH_OPEN, kernel_for_opening, iterations=1)
        close_gray = cv2.morphologyEx(open_gray, cv2.MORPH_CLOSE, kernel_for_closing, iterations=3)
        MIN_AREA = int(config['minimum_size_of_pcb_area'])
        (contours, _) = cv2.findContours(close_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        my_single_contour = None

        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_AREA]
        if len(contours) == 0:
            print('No contour found. Consider a smaller MIN_AREA value.')
            raise Exception
        if len(contours) > 1:
            print('Find more than one contour of PCB area. Consider a larger MIN_AREA value. Using the first contour found.')
            for contour in contours:
                print(f'contour found with area {cv2.contourArea(contour)}.')
        my_single_contour = contours[0]
        # Filter the contour by frequencies on x and y coordinates to get a "box-like" contour
        my_single_contour = filter_contour(my_single_contour)

        # Create a canvas (slightly bigger than the pcb area). Draw pcb on the canvas.
        (upper_left_x, upper_left_y) = my_single_contour.reshape(-1, 2).min(0)
        (lower_right_x, lower_right_y) = my_single_contour.reshape(-1, 2).max(0)
        pcb_width = lower_right_x - upper_left_x
        pcb_heigth = lower_right_y - upper_left_y
        if is_golden_sample:
            canvas_width, canvas_height = int(pcb_width*1.1), int(pcb_heigth*1.1)
        else:
            golden_sample_image = _load_golden_sample_image(config['golden_sample_path'])
            canvas_width, canvas_height = golden_sample_image.shape[1], golden_sample_image.shape[0]
        canvas = 0 * np.ones(shape=[canvas_height, canvas_width, 3], dtype=np.uint8)
        canvas[0:pcb_heigth, 0:pcb_width, :] = image_raw[upper_left_y:lower_right_y, upper_left_x:lower_right_x, :]
        return canvas

def _load_golden_sample_image(path):
        try:
            golden_sample = cv2.imread(path)
            return golden_sample
        except:
            print('Can not find golden sample. Check path of yaml file.')
            raise Exception

def filter_contour(contour):
    
    # Unpack points from contour
    points = [item[0] for item in contour]
    # Convert points to np arrays
    points = np.array(points)
    # Get list of coordinates of each axis
    x_axis = points[:, 0]
    y_axis = points[:, 1]

    # Compute and get the two intervals that contain the most coordinates
    def get_interval_by_axis(axis_points, bins=50):
        # Get histogram data using numpy.histogram
        n, bins = np.histogram(axis_points, bins=bins)
        sort_indices = np.argsort(n)
        bar_delta = bins[1] - bins[0]
        bar_1 = (bins[0] + sort_indices[-1] * bar_delta, bins[0] + (sort_indices[-1] + 1) * bar_delta)
        bar_2 = (bins[0] + sort_indices[-2] * bar_delta, bins[0] + (sort_indices[-2] + 1) * bar_delta)
        max_bars = (  bar_1 ,  bar_2  )
        return max_bars

    # Get the intervals for both axis
    x_intervals = get_interval_by_axis(x_axis)
    y_intervals = get_interval_by_axis(y_axis)

    # Filter the contour points by the intervals
    def filter_contour_by_intervals(points, x_intervals, y_intervals):
        filtered_points = []
        for point in points:
            # If x-coordinate of point is inside any of the x intervals, and y-coordinate of that point does not exceed minumum and maximum of both y intervals
            if   (point[0] > x_intervals[0][0] and point[0] < x_intervals[0][1]) or (point[0] > x_intervals[1][0] and point[0] < x_intervals[1][1]):
                if (point[1] > y_intervals[0][1] and point[1] > y_intervals[1][1]) or (point[1] < y_intervals[0][0] and point[1] < y_intervals[1][0]):
                    continue
                else:
                    filtered_points.append(point)
            # If y-coordinate of point is inside any of the y intervals, and x-coordinate of that point does not exceed minumum and maximum of both x intervals
            elif (point[1] > y_intervals[0][0] and point[1] < y_intervals[0][1]) or (point[1] > y_intervals[1][0] and point[1] < y_intervals[1][1]):
                if (point[0] > x_intervals[0][1] and point[0] > x_intervals[1][1]) or (point[0] < x_intervals[0][0] and point[0] < x_intervals[1][0]):
                    continue
                else:
                    filtered_points.append(point)
        return np.array(filtered_points)

    filtered_points = filter_contour_by_intervals(points, x_intervals, y_intervals)

    # Zip the points into the format of contour points
    filtered_contour = [ [point] for point in filtered_points]
    filtered_contour = np.array(filtered_contour)

    return filtered_contour

def _align_capture_image_with_golden_sample(compared_image, golden_sample_image):
        """align captured pcb area image with golden sample.

        Args:
            compared_image (np.ndarray): captured from camera and processed by _find_pcb_area_of_image
            golden_sample_image (np.ndarray): loaded from disk.
        """
        PIXEL_DIFFERENCE_OF_MATCH_POINTS = 30
        TOP_MATCH_RATE = 0.05

        sift = cv2.xfeatures2d.SIFT_create()
        compared_image_gray = cv2.cvtColor(compared_image, cv2.COLOR_BGR2GRAY)
        golden_sample_image_gray = cv2.cvtColor(golden_sample_image, cv2.COLOR_BGR2GRAY)
        # Display compared images if the flag is on
        SHOW_IMG_FLAG = False
        if SHOW_IMG_FLAG:
            cv2.imshow('gold', golden_sample_image)
            cv2.imshow('comp', compared_image)
            cv2.waitKey(0)
        keypoints_compared_image, description_compared_image = sift.detectAndCompute(compared_image_gray, None)
        keypoints_golden_image, description_golden_image = sift.detectAndCompute(golden_sample_image_gray, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(description_compared_image, description_golden_image, k=1)
        knn_matches.sort(key=lambda x: x[0].distance, reverse=False)
        numGoodMatches = int(len(knn_matches) * TOP_MATCH_RATE)
        pending_top_matches = [x[0] for x in knn_matches[:numGoodMatches]]
        print(f'pending top match qty is {len(pending_top_matches)}')
        good_matches = []
        for match in pending_top_matches:
            x1 = keypoints_compared_image[match.queryIdx].pt[0]
            y1 = keypoints_compared_image[match.queryIdx].pt[1]
            x2 = keypoints_golden_image[match.trainIdx].pt[0]
            y2 = keypoints_golden_image[match.trainIdx].pt[1]
            # the axis of match points should be very very similar/close.
            if abs(x1-x2) < PIXEL_DIFFERENCE_OF_MATCH_POINTS and abs(y1-y2) < PIXEL_DIFFERENCE_OF_MATCH_POINTS:
                good_matches.append(match)
        print(f'Final good matches qty : {len(good_matches)}')
        if len(good_matches) < 10:
            print('Too few good match points.')
            raise Exception
        # -- Localize the object
        obj = np.empty((len(good_matches), 2), dtype=np.float32)
        scene = np.empty((len(good_matches), 2), dtype=np.float32)
        for i in range(len(good_matches)):
            # -- Get the keypoints from the good matches
            obj[i, 0] = keypoints_compared_image[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = keypoints_compared_image[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = keypoints_golden_image[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = keypoints_golden_image[good_matches[i].trainIdx].pt[1]
        H, _ = cv2.findHomography(obj, scene, cv2.RANSAC)
        golden_sample_gray_height, golden_sample_gray_width = golden_sample_image_gray.shape
        compared_image_after_alignment = cv2.warpPerspective(
            compared_image, H, (golden_sample_gray_width, golden_sample_gray_height))
        return compared_image_after_alignment


if __name__ =='__main__':
    path1 = 'raw_images\Picture1.png'
    raw_image = cv2.imread(path1)
    # 调整窗口大小
    cv2.namedWindow("raw_image", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
    cv2.resizeWindow("raw_image", 1600, 900)    # 设置长和宽
    cv2.imshow("raw_image", raw_image)   
    cv2.waitKey (0)  
    cv2.destroyAllWindows()
    canvas = _find_pcb_area_of_image(raw_image)

    cv2.namedWindow("process_img", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
    cv2.resizeWindow("process_img", 1600, 900)    # 设置长和宽
    cv2.imshow("process_img", canvas)   
    cv2.waitKey (0)  
    cv2.destroyAllWindows()
    
    golden_sample_image = _load_golden_sample_image(config['golden_sample_path'])
    compared_image_after_alignment = _align_capture_image_with_golden_sample(raw_image, golden_sample_image)

    cv2.namedWindow("3", 0)  
    cv2.resizeWindow("3", 1600, 900)
    cv2.imshow("3",compared_image_after_alignment)   
    cv2.waitKey (0)  
    cv2.destroyAllWindows()