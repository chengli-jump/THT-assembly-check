import cv2
import numpy as np
import yaml
import sys
sys.path.append('D:\\zhongchengli\\tht-assembly-check')  #加入路径，添加目录

from image_process import check_methods

def _find_pcb_area_of_image(image_raw, is_golden_sample=True, hue_range=(45, 150), saturation_range=(20, 150), value_range=(0, 255)):
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
        cv_show("mask",mask)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(image_raw, image_raw, mask=mask)
        cv_show("res",res)
        hsv_thres = cv2.adaptiveThreshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 0)
        '''
        hsv_thres1 = cv2.adaptiveThreshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 0)
        '''
        cv_show('hsv_thres',hsv_thres)
        #cv_show('hsv_thres1',hsv_thres1)
        kernel_for_opening = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        kernel_for_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        open_gray = cv2.morphologyEx(hsv_thres, cv2.MORPH_OPEN, kernel_for_opening, iterations=1)
        cv_show('open_gray',open_gray)
        close_gray = cv2.morphologyEx(open_gray, cv2.MORPH_CLOSE, kernel_for_closing, iterations=3)
        cv_show('close_gray',close_gray)
        MIN_AREA = int(config['minimum_size_of_pcb_area'])
        (contours, _) = cv2.findContours(close_gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#cv2.CHAIN_APPROX_SIMPLE
        my_single_contour = None
        
        img1 = image_raw.copy()
        img2 = image_raw.copy()
        img3 = image_raw.copy()
        cv2.drawContours(img1,contours,-1,(0,0,255),3) 
        cv_show('img1',img1)

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
        cv_show('img2',img2)
        # Filter the contour by frequencies on x and y coordinates to get a "box-like" contour
        my_single_contour = filter_contour(my_single_contour)


        # Create a canvas (slightly bigger than the pcb area). Draw pcb on the canvas.
        (upper_left_x, upper_left_y) = my_single_contour.reshape(-1, 2).min(0)
        (lower_right_x, lower_right_y) = my_single_contour.reshape(-1, 2).max(0)
        pcb_width = lower_right_x - upper_left_x
        pcb_heigth = lower_right_y - upper_left_y

        finalimg = image_raw[upper_left_y:lower_right_y, upper_left_x:lower_right_x, :]
        cv_show('finalimg',finalimg)
        

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

def cv_show(name,img):
    # 调整窗口大小
    cv2.namedWindow(name, 0)   # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
    cv2.resizeWindow(name, 1600, 900)   # 设置长和宽
    cv2.imshow(name,img)   
    cv2.waitKey (0)  
    cv2.destroyAllWindows()


if __name__ =='__main__':

    file_directory = 'D:\\zhongchengli\\tht-assembly-check\\config\\7-21.yaml'
    with open(file_directory, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    #path1 = 'golden_images\sample1.jpg'
    path2 = r'raw_images\good2_8-20.jpg'
    savepath = 'golden_images\Sample1.jpg'

    #raw_image = cv2.imread(path1)
    #cv_show("raw_image",raw_image)

    raw_image1 = cv2.imread(path2)

    canvas = _find_pcb_area_of_image(raw_image1)
    

    cv_show("process_img",canvas)

    #cv2.imwrite(savepath, canvas)
    #print('ok')