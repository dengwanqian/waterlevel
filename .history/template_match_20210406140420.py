import cv2 as cv
import numpy as np


def template_matching():
    sample = cv.imread(r'images/metertag.jpg')    # 模板图像
    target = cv.imread(r'images/meter1.jpg')    # 待检测图像
    cv.imshow('sample', sample)
    cv.imshow('target', target)
    # 三种模板匹配算法
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    height, width = sample.shape[:2]   # 模板图像的高 宽
    for method in methods:
        print(method)
        result = cv.matchTemplate(image=target, templ=sample, method=method)  # 计算那个区域匹配最好
        # 在匹配的结果中寻找   最小值  最大值及最小、最大值的位置
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if method == cv.TM_SQDIFF_NORMED:   # 如果是标准平方差匹配  取最小值位置
            left_top = min_loc
        else:
            left_top = max_loc
        right_bottom = (left_top[0] + width, left_top[1] + height)  # 加上宽  高
        # 匹配到最佳位置    画小矩形
        cv.rectangle(img=target, pt1=left_top, pt2=right_bottom, color=(0, 0, 255), thickness=2)
        cv.imshow('match-' + np.str(method), target)

template_matching()
cv.waitKey(0)
cv.destroyAllWindows()