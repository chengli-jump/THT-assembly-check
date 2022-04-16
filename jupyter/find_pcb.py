import cv2
import numpy as np
import yaml
import sys
sys.path.append('F:\\tht-assembly-check\\tht-assembly-check')  #加入路径，添加目录

from image_process import check_methods
 
def _find_pcb_area_of_image(image_raw, is_golden_sample=False, hue_range=(45, 150), saturation_range=(20, 150), value_range=(0, 255)):
        hsv_image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2HSV)
        # If config file defines hsv range, use specified range instead.
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
        # cv_show("mask",mask)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(image_raw, image_raw, mask=mask)
        # cv_show("res",res)
        hsv_thres = cv2.adaptiveThreshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 0)
        '''
        hsv_thres1 = cv2.adaptiveThreshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 0)
        '''
        # cv_show('hsv_thres',hsv_thres)
        #cv_show('hsv_thres1',hsv_thres1)
        kernel_for_opening = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        kernel_for_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        open_gray = cv2.morphologyEx(hsv_thres, cv2.MORPH_OPEN, kernel_for_opening, iterations=1)
        # cv_show('open_gray',open_gray)
        close_gray = cv2.morphologyEx(open_gray, cv2.MORPH_CLOSE, kernel_for_closing, iterations=3)
        # cv_show('close_gray',close_gray)
        MIN_AREA = int(config['minimum_size_of_pcb_area'])
        (contours, _) = cv2.findContours(close_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        my_single_contour = None
        
        img1 = image_raw.copy()
        img2 = image_raw.copy()
        img3 = image_raw.copy()
        cv2.drawContours(img1,contours,-1,(0,0,255),3) 
        # cv_show('img1',img1)

        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_AREA]
        if len(contours) == 0:
            print('No contour found. Consider a smaller MIN_AREA value.')
            raise Exception
        if len(contours) > 1:
            print('Find more than one contour of PCB area. Consider a larger MIN_AREA value. Using the first contour found.')
            for contour in contours:
                print(f'contour found with area {cv2.contourArea(contour)}.')
        my_single_contour = contours[0]

        cv2.drawContours(img2,my_single_contour,-1,(0,0,255),3) 
        # cv_show('img2',img2)
        # Filter the contour by frequencies on x and y coordinates to get a "box-like" contour
        my_single_contour = filter_contour(my_single_contour)


        # Create a canvas (slightly bigger than the pcb area). Draw pcb on the canvas.
        (upper_left_x, upper_left_y) = my_single_contour.reshape(-1, 2).min(0)
        (lower_right_x, lower_right_y) = my_single_contour.reshape(-1, 2).max(0)
        pcb_width = lower_right_x - upper_left_x
        pcb_heigth = lower_right_y - upper_left_y

        finalimg = image_raw[upper_left_y:lower_right_y, upper_left_x:lower_right_x, :]
        # cv_show('finalimg',finalimg)

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

        # sift = cv2.xfeatures2d.SIFT_create()
        sift = cv2.SIFT_create()
        compared_image_gray = cv2.cvtColor(compared_image, cv2.COLOR_BGR2GRAY)
        golden_sample_image_gray = cv2.cvtColor(golden_sample_image, cv2.COLOR_BGR2GRAY)
        # Display compared images if the flag is on
        SHOW_IMG_FLAG = False
        if SHOW_IMG_FLAG:
            cv2.imshow('gold', golden_sample_image)
            cv2.imshow('comp', compared_image)
            cv2.waitKey(0)
        #sift.detectAndComputer(gray， None)  # 计算出图像的关键点和sift特征向量
        keypoints_compared_image, description_compared_image = sift.detectAndCompute(compared_image_gray, None)
        keypoints_golden_image, description_golden_image = sift.detectAndCompute(golden_sample_image_gray, None)
        # 获得knn检测器
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(description_compared_image, description_golden_image, k=1)
        knn_matches = list(knn_matches)
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
        #10
        if len(good_matches) < 5:
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
        
def _compare(image, golden_sample_image):
    """compare compared_image_after_alignment with golden_sample_image. 
    Find differences. 

    Args:
        image (np.ndarray): to be compared image after alignment.
        golden_sample_image (np.ndarray): golden sample image
    """
    IMG_BINARY_VAL =config['binary_val']
    print(f'Using IMG_BINARY_VAL : {IMG_BINARY_VAL}')

    # image = automatic_brightness_and_contrast(image, clip_hist_percent=5)

    # Add saturation to images
    saturation_value = int(config['saturation_value'])
    image_saturated = add_saturation(image, saturation_value)
    #step1:调节模板饱和度
    golden_sample_image_saturated = add_saturation(golden_sample_image, saturation_value)
        
    # Get the absolute difference between images(计算区域像素差，白值越高，相似度越差)
    diff_img = cv2.cvtColor(cv2.absdiff(image_saturated, golden_sample_image_saturated), cv2.COLOR_BGR2RGB)
    #
    cv_show('diff_img',diff_img)
    diff_channels = cv2.split(diff_img)
    channel_sequence = {'red': 0, 'green': 1, 'blue': 2}
    check_pass_flag = True
    # Get a copy of the original picture for drawing check-boxes
    result_image = np.copy(image)
    for each_check_box in config['check_box']:
        # get value from config
        check_name = each_check_box['name']
        box_upper_left = eval(each_check_box['box_upper_left'])
        box_lower_right = eval(each_check_box['box_lower_right'])
        check_channel = diff_channels[channel_sequence[each_check_box['check_channel']]]
        threshold_method = each_check_box['threshold_method']
        threshold_value = each_check_box['threshold_value']
            
        if threshold_method == "by_percent":
            # binary box_area and get white val
            rect_area = check_channel[box_upper_left[1]:box_lower_right[1], box_upper_left[0]:box_lower_right[0]]
            _, binary_rect_area = cv2.threshold(rect_area, IMG_BINARY_VAL, 255, cv2.THRESH_BINARY)
            #cv_show('diff',binary_rect_area)
            white_total = np.count_nonzero(binary_rect_area == 255)
            black_total = np.count_nonzero(binary_rect_area == 0)
            white_percent = round(white_total/(white_total+black_total)*100, 2)

        else:
            box_slice_image_golden = golden_sample_image[box_upper_left[1]:box_lower_right[1], box_upper_left[0]:box_lower_right[0], :]
            box_slice_image = image[box_upper_left[1]:box_lower_right[1], box_upper_left[0]:box_lower_right[0], :]
            white_total, white_percent = check_methods.check_by_method(box_slice_image, box_slice_image_golden, threshold_method)

        # compare with threshold
        box_area_unmatched = False if white_percent < threshold_value else True

        # Write results to log
        # self.my_logger.debug(f'{"FAIL" if box_area_unmatched else "PASS"}: {check_name} - white_total: {white_total}')
        # self.my_logger.debug(f'{check_name} - black_total: {black_total}')
        print(f'{"FAIL" if box_area_unmatched else "PASS"}: {check_name} - white_pecent: {white_percent} %')    


        if box_area_unmatched:
            cv2.rectangle(result_image, box_upper_left, box_lower_right, (0, 0, 255), 2)  # red box
            check_pass_flag = False
        else:
            cv2.rectangle(result_image, box_upper_left, box_lower_right, (0, 255, 0), 2)  # green box
        mark_string = 'White:' + str(white_total) + '(' + str(white_percent) + '%)'
        cv2.putText(result_image, mark_string, box_upper_left, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return(check_pass_flag, result_image)

def add_saturation(image, saturation):

    BASE_VALUE = 100
    hlsImg = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2HLS) 
    #H-色调，L-亮度，S-饱和度
    # 饱和度
    hlsImg[:, :, 2] = (1.0 + saturation / float(BASE_VALUE)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    lsImg = lsImg.astype(np.uint8)
    return lsImg

def cv_show(name,img):
    # 调整窗口大小
    cv2.namedWindow(name, 0)   # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
    cv2.resizeWindow(name, 1600, 900)   # 设置长和宽
    cv2.imshow(name,img)   
    cv2.waitKey (0)  
    cv2.destroyAllWindows()

if __name__ =='__main__':

    file_directory = 'F:\\tht-assembly-check\\tht-assembly-check\config\\7-21.yaml'
    with open(file_directory, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    path1 = r'raw_images\Picture1.png'
    raw_image = cv2.imread(path1)
    # cv_show("raw_image",raw_image)
    canvas = _find_pcb_area_of_image(raw_image)
    

    # cv_show("process_img",canvas)

    
    golden_sample_image = _load_golden_sample_image(config['golden_sample_path'])
    cv_show("sample_image",golden_sample_image)

    compared_image_after_alignment = _align_capture_image_with_golden_sample(canvas, golden_sample_image)
    cv_show("compared_image",compared_image_after_alignment)

    check_pass, final_image = _compare(compared_image_after_alignment, golden_sample_image)

    cv_show("final_image",final_image)

