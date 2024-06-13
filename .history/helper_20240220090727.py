import cv2
import numpy as np
import math

isshow=True # By default, show image in window, if you want to save image, set isshow=False

# show image or save image
def imshow(title,image):
    if image is None: return
    if isshow:
        cv2.imshow(title,image)
    else:
        cv2.imwrite("./out/{title+".jpg",image)
        pass

# guess ruler lines
def guessMeterLines(image,params):
    img=image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(image, (5, 5), 0)
    th0 = cv2.adaptiveThreshold(gray, 256, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9 , \
                                params['adaptiveoffset'])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th1=th0
    th1 = cv2.erode(th1, kernel, iterations=1) # with no erode and dilate
    th1 = cv2.dilate(th1, kernel, iterations=1)
    th1 = cv2.erode(th1,kernel,iterations=1)

    #imshow("line", th1)
    edge = cv2.Canny(th1, 50, 150)
    #imshow("edge", edge)
    minLineLength = 25  # height/32
    maxLineGap = params["maxLineGap"]  # height/40
    maxcos=0.4
    if 'maxcos' in params:
        maxcos=params['maxcos']
    lines = cv2.HoughLinesP(th1, 1, np.pi / 180, 90, minLineLength, maxLineGap)
    if lines is None:
        return img, []
    x0 = 0
    y0 = 0
    w = 0
    h = 0
    roi_lines=[]

    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x2-x1)/math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))<maxcos  and x1>th1.shape[1]//100 \
                    and x1<th1.shape[1]*14//15 \
                    and y2>th1.shape[0]//8 and y2<th1.shape[0]*7//8:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                roi_lines.append((x1, y1, x2, y2))

    if len(roi_lines)>0: # exclude fake lines
        mean=np.mean(roi_lines,0)[0]
        std=np.std(roi_lines,0)[0]
        for line in roi_lines[:]:
            if abs(line[0]-mean)>std+50 or (line[1]>image.shape[0]*3/4 and line[3]>image.shape[0]*3/4):
                roi_lines.remove(line)
                #print("remove line",line)
    roi_lines.sort(key=lambda line: (line[1])) # let the lines over water in the front

    return img, roi_lines

#guess ruler positions
def guessMeterPositions(image,template,params):

    img = image.copy()

    img1 = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[128, 128, 128])
    source_h,source_w,_=img1.shape
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    th0 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9, params['adaptiveoffset']) #2,5
    th1=th0

    # open and close operation to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params['erodekernel'], params['erodekernel']))

    th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    th1 = cv2.erode(th1, kernel, iterations=params['prehandle']+1)
    th1 = cv2.dilate(th1, kernel, iterations=params['prehandle'])
    
    edge = cv2.Canny(th1, 50, 150,apertureSize= params['apertureSize'])
    imshow('roitag edge',edge)
    binary,contours,_ = cv2.findContours(
        edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda c: np.min(c[:, :, 0]))

    template = cv2.copyMakeBorder(template, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.GaussianBlur(template, (5, 5), 0)
    template = cv2.adaptiveThreshold(template, 230, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9, params['adaptiveoffset'])
    edge_template=cv2.Canny(template, 50, 150,apertureSize= params['apertureSize'])
    binary,contours_template, _ = cv2.findContours(
        edge_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_template=None

    for cnt in range(len(contours_template)):
        if cv2.contourArea(contours_template[cnt])>150:
            cnt_template=contours_template[cnt]

    roi_list = []

    for cnt in range(len(contours)):
        area = cv2.contourArea(contours[cnt])
        if area > params['minareasize'] and area<2000:
            # fetch the external rectangle
            x, y, w, h = cv2.boundingRect(contours[cnt])

            if w > h or w*2<h or y <source_h//8 or y+h >source_h*7//8 or w<18:
                continue
            match_index=cv2.matchShapes(contours[cnt], cnt_template, 1, 0.0)
            if match_index<params['maxshapeindex']:
                roi_list.append((x,y,w,h,match_index))
                cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(img1, [contours[cnt]], 0, (255, 0, 255), 2)
    if len(roi_list)>0:
        roi_list.sort(key=lambda roi: (roi[1]))
        mean=np.mean(roi_list,0)[0]
        std=np.std(roi_list,0)[0]
        for roi in roi_list[:]:
            if abs(roi[0]-mean)>2*std and len(roi_list)>4:
                roi_list.remove(roi)

    return img1,roi_list

# judge if a jpg file is valid
def is_valid_jpg(jpg_file):
    with open(jpg_file, 'rb') as f:
        f.seek(-2, 2)
        buf = f.read()
        f.close()
        return buf ==  b'\xff\xd9' 


