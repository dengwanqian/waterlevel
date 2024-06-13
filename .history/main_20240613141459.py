# Read grids of a water ruler,which has been tested with opencv-python 3.4.13

import cv2
import numpy as np
import math
import datetime
import helper
import os
import math
import copy

# Properties of the water ruler image file, template tag，and parameters of the algorithms to guess the water level  
improps_init={
    'imagepath':'./images',

    'fileprefix':'meter2',
    #'fileprefix':'meter5',
    'filetype':'jpg',
    'blurvalue':0,
    'threshold_val' :100,
    'threshold_plus' : 0,
    'tagposition':(0,0,0,0),
    'the_tag':None,
    'path':'',
    'templatepath':'template/metertag.jpg',
    'guesslineparams':
        [{'minLineLength':32,
        'maxLineGap':15,  #25,30
        'adaptiveoffset':1, #1,2
        'maxcos':0.4},
         {'minLineLength':26,
        'maxLineGap':5,  #25,30
        'adaptiveoffset':1, #1,2
        'maxcos':0.4},
        {'minLineLength':28,
        'maxLineGap':6,  #25,30
        'adaptiveoffset':1, #1,2
        'maxcos':0.4}],
    'guessroitagparams':
        [{
            'apertureSize':7,
            'prehandle':2,
            'erodekernel':3,
            'adaptiveoffset':2,
            'maxshapeindex':1.5,#1.5
            'minareasize':150},
        {
            'apertureSize':7,
            'prehandle':1,
            'erodekernel':2,
            'adaptiveoffset':2,
            'maxshapeindex':1.5,
            'minareasize':150},
        {
            'apertureSize':7,
            'prehandle':0,
            'erodekernel':2,
            'adaptiveoffset':2,
            'maxshapeindex':1.5,
            'minareasize':150
        }],
    'guessgridparams':
		[{
            'gaussianblurbox': 5,
            'kernelbox': 2,
            'kernel1box': 2,
            'erodetimes': 2,
            'leftpad': 2,
            'rightpad': 0
        },
		{
            'gaussianblurbox': 5,
            'kernelbox': 2,
            'kernel1box': 2,
            'erodetimes': 2,
            'leftpad': 3,
            'rightpad':1
        },
		{
            'gaussianblurbox': 5,
            'kernelbox': 2,
            'kernel1box': 3,
            'erodetimes': 2,
            'leftpad': 3,
            'rightpad':3
        },
		{
            'gaussianblurbox': 5,
            'kernelbox': 3,
            'kernel1box': 2,
            'erodetimes': 2,
            'leftpad': 3,
            'rightpad':6
        },
		{
            'gaussianblurbox': 5,
            'kernelbox': 5                                                                                                                                                                                                 ,
            'kernel1box': 2,
            'erodetimes': 2,
            'leftpad': 16,
            'rightpad':1
        },
        {
            'gaussianblurbox': 5,
            'kernelbox': 7,
            'kernel1box': 2,
            'erodetimes': 2,
            'leftpad': 6,
            'rightpad': 6
            }
        ]
}

# to guess the position of the water ruler in the image.
# and fathermore locate the 
def locateMeter(improps):
    img = cv2.imread(improps['path'])
    origin=img.copy()

    image = img.copy()
    params2=improps['guesslineparams'][0]
    
    #guess meter lines，using several groups of parameters to guess.

    image2,roi_lines=helper.guessMeterLines(img,params2)
    helper.imshow(improps['fileprefix'] + "_guessMeterLines_init1", image2)

    if len(roi_lines)<=4 or  (len(roi_lines)==2 and abs(roi_lines[0][0]-roi_lines[1][0]))>150 or len(roi_lines)>0 and np.max(roi_lines,0)[3]<image2.shape[0]/8 \
            or len(roi_lines)>0 and np.min(roi_lines,0)[3]>image2.shape[0]*7/8:
        params2=improps['guesslineparams'][1]
        image2,roi_lines2=helper.guessMeterLines(img,params2)
        helper.imshow(improps['fileprefix'] + "_guessMeterLines_init2", image2)
        if len(roi_lines2)>=len(roi_lines):
            roi_lines=roi_lines2
    if len(roi_lines)<=4 or  (len(roi_lines)==2 and abs(roi_lines[0][0]-roi_lines[1][0]))>150 or len(roi_lines)>0 and np.max(roi_lines,0)[3]<image2.shape[0]/8 \
            or len(roi_lines)>0 and np.min(roi_lines,0)[3]>image2.shape[0]*7/8:
        params2=improps['guesslineparams'][2]
        image2,roi_lines3=helper.guessMeterLines(img,params2)
        helper.imshow(improps['fileprefix'] + "_guessMeterLines_init3", image2)
        if len(roi_lines2)>=len(roi_lines):
            roi_lines=roi_lines2
    print(params2)
    print('roi_lines',roi_lines)

    # judge if it is needed to 
    #need rotate?
    
    line_mean=0
    degree=0
    if len(roi_lines)>1:
        line_mean=np.mean(roi_lines,0)[0]
        line_std=np.std(roi_lines,0)[0]
        line_max=np.max(roi_lines,0)[0]
        line_min=np.min(roi_lines,0)[0]
        x=np.mean(roi_lines,0)[2]-np.mean(roi_lines,0)[0]
        cos_sum=0
        cos=0
        num=len(roi_lines)
        if abs(np.mean(roi_lines,0)[2]-np.mean(roi_lines,0)[0])>0:
            i=0
            for line in roi_lines[:]:
                x=line[2]-line[0]
                y=line[3]-line[1]
                if y>0:x=-x
                z=math.sqrt(x*x+y*y)

                # drop the line under the water level.
                if cos_sum*x>=0 or (line[3]<image2.shape[0]*2/3 and len(roi_lines)<4) : 
                    cos_sum = cos_sum + x / z
                else:
                    roi_lines.remove(line)
                    num=num-1
                if x==0 and i%3==0:
                    i += 1
                    num=num-1
            cos=cos_sum/num
            print('cos',cos)
        else:
            cos=0

        inv=np.arccos(cos)
        print('inv',inv)
        degree=90-np.degrees(inv)
        if degree>0 and cos<0: degree=-degree
        improps['degree']=degree
        print('degree',degree)

    if abs(degree)>0.8 : 
        # if the degree is larger than 0.8, rotate the image.
        img=image.copy()

        (h, w) = img.shape[:2]
        (cX, cY) = (round(line_mean), round(h / 2))
        M = cv2.getRotationMatrix2D((line_mean,h/2), degree, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        img = cv2.warpAffine(image, M, (nW, nH))
        image=img.copy()

        params2['maxcos']=0.1 # already rotated, so the maxcos should be smaller.
        params2['adaptiveoffset']=2
        image2,roi_lines=helper.guessMeterLines(img,params2);
 
    #continue to exclude the non vertical lines and the lines too above in the image.
    for line in roi_lines[:]:
        if abs(line[2]-line[0])>5:
            roi_lines.remove(line)
            print('remove0:', line)
        elif (line[1]<image2.shape[0]/5 or line[3]<image2.shape[0]/5) and len(roi_lines)>3 :
            roi_lines.remove(line)
            print('remove1:', line)
        elif len(roi_lines)>3 and abs(line[3]-line[1])<22:
            roi_lines.remove(line)
            print('remove2:', line)

    #continue to exclude lines far away from the mean of the lines.
    if len(roi_lines)>4:
        line_mean=np.mean(roi_lines,0)[0]
        line_std=np.std(roi_lines,0)[0]
        line_max=np.max(roi_lines,0)[0]
        line_min=np.min(roi_lines,0)[0]
        for line in roi_lines[:]:
            if abs(line[0]-line_mean)>line_std*3:
                roi_lines.remove(line)
                print('remove4:',line)

    improps['roi_lines']=roi_lines
    print('roi_lines',roi_lines)
    helper.imshow(improps['fileprefix']+"_guessMeterLines",image2)

    #find meter tag
    template=cv2.imread(improps['templatepath'])

    params=improps['guessroitagparams'][0]

    image,roi_list=helper.guessMeterPositions(image,template,params)
    helper.imshow('roi_list1',image)
    print('roi_list1',roi_list)
    roi_list0=roi_list
    if len(roi_list)<=3:
        params=improps['guessroitagparams'][1]
        image,roi_list=helper.guessMeterPositions(image,template,params)
        helper.imshow('roi_list2', image)
        print('roi_list2', roi_list)
        if len(roi_list)>len(roi_list0):
            roi_list0=roi_list
    if len(roi_list)<=3:
        params=improps['guessroitagparams'][2]
        image,roi_list=helper.guessMeterPositions(image,template,params)
        helper.imshow('roi_list3', image)
        print('roi_list3', roi_list)
        if len(roi_list)>len(roi_list0):
            roi_list0=roi_list
    roi_list=roi_list0
    helper.imshow(improps['fileprefix']+"_guessMeterPossions",image)

    the_roi=None
    left=10000
    right=0

    #combine the positions of the lines and the positions of the 'E' tags to judge position of the ruler.
    line_mean=0
    line_std=0
    line_max=0
    line_min=0

    for (x1, y1, x2, y2) in roi_lines[:]:
        if abs(x1 - x2) > 5:
            roi_lines.remove((x1, y1, x2, y2))
    if len(roi_lines)>1:
        line_mean=np.mean(roi_lines,0)[0]
        line_mean_y=np.mean(roi_lines,0)[1]
        line_std=np.std(roi_lines,0)[0]
        line_max=np.max(roi_lines,0)[0]
        line_min=np.min(roi_lines,0)[0]

    #according the double vetical lines,remove the tags not in the ruler.
    if len(roi_lines)>=2 and line_std<100 and line_max-line_min>15:
        for roi in roi_list[:]:
            if abs(roi[0]-line_mean)>100 or roi[0]<line_min-2 or roi[0]>line_max+2:
                roi_list.remove(roi)

    improps['roitaglist']=roi_list
    tagposition=(0,0,0,0)
    print('roi_list adjusted',roi_list)
    roi_list2=roi_list.copy()
    if len(roi_list2)>1 and roi_list[0][2]*roi_list[0][2]*0.8<=roi_list[1][2]*roi_list[1][2]:   
        # use the second roi to judge the grid area.
        roi_list2.remove(roi_list2[0])
    if len(roi_list2)>1 and roi_list[0][2]*roi_list[0][2]*0.8<=roi_list[1][2]*roi_list[1][2]:    
        # use the third roi to judge the grid area.
        roi_list2.remove(roi_list2[0])
    for roi in roi_list2:
        x,y,w,h,_=roi
        for roi_line in roi_lines:
            x1,y1,x2,y2=roi_line
            if x1-2<=x and x1+4*w>=x and (left==10000 or left<x1):
                left=x1

            if x1>=x+w-7 and x1-5*w<=x and (right==0 or right>x1):
                right=x1

            if left<10000 and right>0:
                left=left
                right=right
                continue
        if right>0 and left<10000:
            if (right-left)<2*w*0.8:
                left=x-w
            if right<x+w-2:
                right=x+w
            if y+h>img.shape[0]*3/5 and right-left<80 :
                y=y-6*h #handle the tag under the water level.
            
            if abs(left-(x+w/4))<abs(right-(x+w/4)):  #change to get the right side of the ruler.
                x=x+w

            if y>img.shape[0]/2:
                the_roi = img[y - h:y + h, x:x + w]
                tagposition=(x,y-h,w,h)
            else:
                the_roi = img[y :y + 2*h, x:x + w]
                tagposition = (x, y +h, w, h)
            left=x-w
            right=min(x+w+2,right)
            break


    #no tags found, then only use the lines to judge the position of the ruler.
    if (right==0 or left==10000) and len(roi_list)==0:

        if len(roi_lines)>=2:
            left=line_min
            right=line_max
            line_mean_y=int(line_mean_y)
            if right-left>30:
                the_roi = img[line_mean_y:int(line_mean_y+(right-left)*1.3),
                          left + (right - left) // 2:left+(right - left) // 2+min((right - left)//2,25)]

            else:
                the_roi = img[line_mean_y:int(line_mean_y+(right-left)*1.3),
                          left + (right - left) // 2:left+min(right - left,25)]

            tagposition = (left + (right - left) // 2,line_mean_y, min((right-left),25),int((right-left)/2*1.3))

            if right-left<20:
                right=0
                left=10000

    #no lines found, then only use the tags to judge the position of the ruler.
    if (right==0 or left==10000) and len(roi_list)>0:
        roi0 = roi_list[0]
        distance=1000
        if len(roi_lines)==1:
            for roi in roi_list:
                if (roi_lines[0][0]-roi[0])<distance or (roi_lines[0][2]-roi[0])<distance:
                    distance=min(roi_lines[0][0]-roi[0],roi_lines[0][2]-roi[0])
                    roi0=roi

        for roi in roi_list:
            if roi[2]*roi[3]>roi0[2]*roi0[3] and roi[2]*roi[3]<1500:
                roi0=roi
        left=roi0[0]
        right=roi0[0]+roi0[2]
        for roi in roi_list:
            if roi0[0]+roi0[2]//2<roi[0] and roi0[0]+roi0[2]*2>roi[0]:
                #left=roi0[0]
                right=roi[0]+roi[2]

                break

            elif roi[0]+roi[2]//2<=roi0[0] and roi[0]+roi[2]*2>roi0[0]:
                left=roi[0]
                #right=roi0[0]+roi0[2]
                break
        if left==roi0[0] and right==roi0[0]+roi0[2]:
            left=max(roi0[0]-roi0[2],0)
            right=roi0[0]+2*roi0[2]
        (x, y, w, h, _) = roi0
        the_roi=img[y-h:y+h,x:x+w] 
        tagposition=(x,y-h,w,h)
    
    #begin to fetch ruler area.
    if right>0 and left<10000 and len(the_roi)!=0: 
        improps['the_tag']=the_roi
        improps['tagposition']=tagposition
        print('ruler area No.0',0,img.shape[0],left,right)

        image=img[0:img.shape[0],left:right]
        improps['left']=left;
        improps['right']=right;

        #calculate the mean of the tag image.
        the_roi=cv2.cvtColor(the_roi, cv2.COLOR_BGR2GRAY)
        (the_roi_mean,the_roi_std)= cv2.meanStdDev(the_roi)
        helper.imshow('roi_tag',the_roi)

        print('the_roi_mean',the_roi_mean)
        print('the_roi_std',the_roi_std)
        improps['the_tag_mean']=int(the_roi_mean)
        improps['the_tag_std']=int(the_roi_std)

        threshold_val=int(the_roi_mean)
        threshold_plus=int(the_roi_std)

        helper.imshow(improps['fileprefix']+'_result',image)
        img1 = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[32, 32, 32])
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # according the mean of the tag image, adjust the threshold value, to get the best result.the data below is from the test.
        if  threshold_val>190:
            local_threshold_val = threshold_val + threshold_plus * (255-threshold_val)*2.0/ 255
        if  threshold_val>160:
            local_threshold_val = threshold_val + threshold_plus * (255-threshold_val)*1/ 255
        elif threshold_val > 150:
            local_threshold_val = threshold_val + threshold_plus * (255 - threshold_val) * 0.5 / 255 
        elif threshold_val > 130:
            if threshold_plus>40 and threshold_plus<=50:
                local_threshold_val = threshold_val + threshold_plus * (255 - threshold_val) * 0.6 / 255
            else:
                local_threshold_val = threshold_val + threshold_plus * (255 - threshold_val) * 1.2 / 255
        elif threshold_val > 120:
            local_threshold_val = threshold_val + threshold_plus * (255 - threshold_val) * 1.2 / 255
        elif threshold_val > 95:
            local_threshold_val = threshold_val + threshold_plus * (255 - threshold_val) * 1.0/ 255
        elif threshold_val > 85:
            if threshold_plus>40 and threshold_plus<50:
                local_threshold_val = threshold_val + threshold_plus * (255 - threshold_val) * 0.2 / 255
            else:
                local_threshold_val = threshold_val + threshold_plus * (255 - threshold_val) * 0.2/ 255
        elif threshold_val > 75:
            if threshold_plus>40 and threshold_plus<50:
                local_threshold_val = threshold_val + threshold_plus * (255 - threshold_val) * 0.2 / 255
            elif threshold_plus>20 and threshold_plus<30:
                local_threshold_val = threshold_val + threshold_plus * (255 - threshold_val) * 0 / 255
            else:
                local_threshold_val = threshold_val + threshold_plus *(255 - threshold_val) * 1/ 255
        elif threshold_plus<30 and threshold_plus>=24 and threshold_val>60 and  threshold_val<70:
            local_threshold_val = threshold_val + threshold_plus *1.1
        elif threshold_plus<30 and threshold_plus>=24 and threshold_val>=70 and  threshold_val<80:
            local_threshold_val = threshold_val + threshold_plus *0.5
        elif threshold_plus>10 and  threshold_plus<20 and threshold_val>=60 and  threshold_val<70: #for meter7
            local_threshold_val = threshold_val + threshold_plus * 0
        elif threshold_plus>=10 and  threshold_plus<20 and threshold_val>40 and  threshold_val<=50:
            local_threshold_val = threshold_val + threshold_plus * (255 - threshold_val) * (-0.0)/ 255
        elif threshold_plus>60 and  threshold_plus<70 and threshold_val>130 and  threshold_val<140 :
            local_threshold_val = threshold_val + threshold_plus * (threshold_val+50)/ 255
        elif threshold_plus>=20 and  threshold_plus<=25 and threshold_val>=50 and  threshold_val<=55:
            local_threshold_val = threshold_val + threshold_plus * 0.1
        elif threshold_plus>=10 and  threshold_plus<20 and threshold_val>=60 and  threshold_val<70:
            local_threshold_val = threshold_val + threshold_plus * 0
        elif threshold_plus<10  and threshold_val>=40 and  threshold_val<=60:
            local_threshold_val = threshold_val + threshold_plus * 0.53
        else:
            local_threshold_val = threshold_val + threshold_plus * 1.1

        ret, th1 = cv2.threshold(gray, local_threshold_val, 255, cv2.THRESH_BINARY)
        helper.imshow('roi_area0',th1)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        th1 = cv2.erode(th1,kernel1,iterations=1)
        th1 = cv2.dilate(th1, kernel, iterations=2)
        erodetimes=1
        if threshold_plus<15: 
            erodetimes=2
        th1 = cv2.erode(th1,kernel,iterations=erodetimes)

        helper.imshow('roi_area',th1)

        # check the edge

        edge = cv2.Canny(th1, 150, 300,apertureSize=5)

        return origin, img1, edge,improps
    else:
        print(improps['path'],"can't find the ruler!")
        return None,None,None,(0,0)

# fetch rois(regions of interest) according to the contours checked.
def getRoi(img, binary):
    '''
    img: original image
    binary: the canny edge of the image after preprocessed.
    '''

    # find contours
    binary,contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    roi_list = []
    # judge the contours
    area_before = (0, 0, 0, 0)
    rotated = False
    for cnt in range(len(contours)):
        area = cv2.contourArea(contours[cnt])
        # judge and fetch contours needed. the area should be larger than 2500 according to the test.
        if area > 2500:
            # fetch the outer rectangle of the contour
            x, y, w, h = cv2.boundingRect(contours[cnt])

            if w * 2 > h:
                continue
                # pass

            area_now = (x, y, w, h)
            if area_now != area_before:
                roi_list.append(area_now)
                area_before = area_now
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(img, [contours[cnt]], 0, (255, 0, 255), 2)
        roi_list.sort(key=lambda oneroi: np.min(oneroi[1]))

    return img, roi_list


# Save  roi
def saveRoi(src, roi_list):
    '''
    src: copy of the original image 
    roi_list: List of roi
    '''

    count = 0
    if len(roi_list) == 0 or roi_list[-1] == (0, 0, 0, 0):
        return count
    for i in range(len(roi_list)):
        count += 1
        x, y, w, h = roi_list[i]
        roi = src[y:y + h, x:x + w]
    return count

# process roi area
def  processRoi(roi_area,param):
    '''
    roi_area: roi area
    param: parameters of the algorithms to guess the water level
    '''
    fileprefix=param['fileprefix']
    h, w, _ = roi_area.shape
    if param['kernelbox']>5:
        param['rightpad']=max(param['rightpad'],w/4)

    if w/2-param['leftpad']-param['rightpad']<10 and param['rightpad']>=10:
        param['rightpad']=6
    if w/2-param['leftpad']-param['rightpad']<10 and param['leftpad']>10:
        param['leftpad']=6
    img1 = roi_area[0:h - 2, round(w / 2+param['leftpad']):round(w-param['rightpad'])]
    img1 = cv2.copyMakeBorder(img1,10, 10, 10, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)

    helper.imshow(fileprefix+"roi_area_adjust", img1)
    print(param)

    # gaussian blur operation
    threshold_plus=param['threshold_plus']
    gray = cv2.GaussianBlur(gray, (param['gaussianblurbox'], param['gaussianblurbox']), 0)

    # set applopriet value for thresholding operation, based on the mean and std of the roi area,and the test data.
    if threshold_plus<10:
        localthresholvalue = param['localthresholvalue'] - threshold_plus * 32 // 15  
    elif threshold_plus<20:
        if param['localthresholvalue']<60:
            localthresholvalue = param['localthresholvalue'] - threshold_plus * 25// 15  
        elif param['localthresholvalue']<110 and param['localthresholvalue']>100:
            localthresholvalue=param['localthresholvalue']-threshold_plus*20//15
        else:
            localthresholvalue=param['localthresholvalue']-threshold_plus*6//15  
    elif threshold_plus<30:
        if param['localthresholvalue']<100:
            localthresholvalue=param['localthresholvalue']-threshold_plus*10//15
        elif param['localthresholvalue']<90:
            localthresholvalue=param['localthresholvalue']-threshold_plus*10//15
        else:
            localthresholvalue = param['localthresholvalue'] - threshold_plus * 11 // 15
    elif threshold_plus<40:
        localthresholvalue=param['localthresholvalue']-threshold_plus*13//15
    elif threshold_plus<50:
        if param['localthresholvalue']>180:
            localthresholvalue=param['localthresholvalue']-threshold_plus*6//15
        elif param['localthresholvalue']>120 and  param['localthresholvalue']<130:
            localthresholvalue = param['localthresholvalue'] - threshold_plus * 25 // 15
        else:
            localthresholvalue = param['localthresholvalue'] - threshold_plus * 10 // 15
    elif threshold_plus<60:
        localthresholvalue=param['localthresholvalue']-threshold_plus*10//15
    elif threshold_plus<70:
        if param['localthresholvalue']>190:
            localthresholvalue=param['localthresholvalue']-threshold_plus*10//15 
        elif param['localthresholvalue']<170 and param['localthresholvalue']>160:
            localthresholvalue = param['localthresholvalue'] - threshold_plus * 1// 15  
        elif param['localthresholvalue']<130 and param['localthresholvalue']>120:
            localthresholvalue = param['localthresholvalue'] - threshold_plus * 10// 15  
        elif param['localthresholvalue']<100 and param['localthresholvalue']>90:
            localthresholvalue = param['localthresholvalue'] - threshold_plus * 10// 15  
        else:
            localthresholvalue=param['localthresholvalue']-threshold_plus*11//15 
    elif threshold_plus<80:
        if param['localthresholvalue']<120:
            localthresholvalue=param['localthresholvalue']-threshold_plus*(-8)//15
        else:
            localthresholvalue = param['localthresholvalue'] - threshold_plus * 10 // 15
    elif threshold_plus<90 :
        localthresholvalue = param['localthresholvalue'] - threshold_plus * 10// 15
    else:
        localthresholvalue = param['localthresholvalue'] - threshold_plus * 14 // 15

    ret, th1_level = cv2.threshold(gray,localthresholvalue, 255, cv2.THRESH_BINARY_INV)

    helper.imshow(fileprefix+"th0_level"+str(h), th1_level)

    # open and close operation to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (param['kernelbox'], param['kernelbox']))

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (param['kernel1box'],param['kernel1box']))

    th1_level = cv2.morphologyEx(th1_level, cv2.MORPH_CLOSE, kernel)
    th1_level = cv2.morphologyEx(th1_level, cv2.MORPH_OPEN, kernel1)

    if param['kernelbox']<5:
        pass
    else:
        th1_level = cv2.dilate(th1_level, kernel, iterations=param['erodetimes'])


    # edge check
    edge_level = cv2.Canny(th1_level, 50, 150)
    helper.imshow(fileprefix+"th1_level"+str(h), th1_level)
    helper.imshow(fileprefix+"edge_level"+str(h), edge_level)

    if param['kernelbox']<5:
        edge_level = cv2.adaptiveThreshold(edge_level, 255, \
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,1) #2,5
    else:
        edge_level=th1_level

    helper.imshow(fileprefix+"edge_level" + str(h), edge_level)

    return th1_level, roi_area, edge_level


# caculate the grids of the ruler.
def getlevel(roi_area, edge_level,param):
    '''
    roi_area:original roi area
    edge_level: the canny edge of the roi area
    '''
    # look for contours
    binary,contours, _ = cv2.findContours(
        edge_level, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    level_list = []
    # judge the contours
    for cnt in range(len(contours)):
        area = cv2.contourArea(contours[cnt])
        # judge and fetch contours needed.
        x, y, w, h = cv2.boundingRect(contours[cnt])

        if w*h > param['tagposition'][2]*param['tagposition'][3]/4.5  and area < 4000 and h>w*1.2:
            # get the outer rectangle of the contour

            area_now = (x, y, w, h)
            level_list.append(area_now)

    return roi_area, level_list

# guess the grids of the ruler.
def guessgrid(maxroi, level_list,param):
    e_mean_hight = 0
    count = 0
    for i in range(len(level_list)):
        if i >= 1 and i < len(level_list)-1 :
            count += 1
            e_mean_hight += level_list[i][3]
    if param['kernelbox'] < 5:
        index=24 /30
    elif param['kernelbox']< 7:
        index=22/30
    else:
        index = 25 / 30
    if count > 0:
        e_mean_hight = round(e_mean_hight / count*index) #adjust the mean hight of the level
    print('e_mean_height',e_mean_hight)
    if e_mean_hight==0 and len(level_list)>=1:
        e_mean_hight=level_list[-1][3]*25/30 
    if len(level_list)>2 and level_list[0][1]-level_list[1][1]<e_mean_hight*1.5:
        level_list.remove(level_list[0])
    level = len(level_list)
    if len(level_list) > 1 and level_list[-1][3] > 2.5 * e_mean_hight and level_list[-1][1] < 15 or \
            len(level_list) > 1 and level_list[-1][3]< e_mean_hight*0.8 or len(level_list) > 1 and level_list[-1][1] < 15 or \
            len(level_list)>1  and level_list[-2][1]-level_list[-1][1]-level_list[-1][3]<level_list[-2][3]/3.5:  # 去掉帽子
        level = level - 1
        level_list.remove(level_list[-1])


    adjust_grid = 0
    if level > 1 and e_mean_hight > 0:
        if level_list[0][3] *1.1 < e_mean_hight:
            level = level - 1
            adjust_grid = int((level_list[0][3] / e_mean_hight) * 5) + 5
        elif level_list[0][3] > int(e_mean_hight * 1.6):
            adjust_grid = int((level_list[0][3] - e_mean_hight) / e_mean_hight * 5)
        else:
            adjust_grid = int((maxroi[3] - (level_list[0][1] +level_list[0][3]-10)) / e_mean_hight * 5)
    for onelevel in level_list:
        if onelevel[2] * onelevel[3] > 6000:
            level = 0
            break
    grid = level * 10 + adjust_grid
    print('grids:', grid)
    return grid

# read the water level of the ruler.
def readWaterLevel(improps):
    fileprefix=improps['fileprefix']
    # read image and locate the meter    
    origin, img, edge,improps= locateMeter(improps)
    # copy image
    src=None
    if img is not None:
        src = img.copy()
    # get roi
    img, roi_list = getRoi(img, edge)
    print('roi_area_lists',roi_list)

    #choose the most suitable roi
    maxroi=None
    if len(roi_list)>0:
        roi_list.sort(key=lambda roi: (roi[1]),reverse=True) #fetch inner contours first
        maxroi=roi_list[0]
        for oneroi in roi_list:
            if  improps['tagposition'][1]>oneroi[1] and  improps['tagposition'][1]<oneroi[1]+oneroi[3]:
                maxroi=oneroi
                break
    if maxroi:
        improps["top"]=maxroi[1]-10
        improps["bottom"]=maxroi[1]+maxroi[3]-10
    helper.imshow( fileprefix+'_meter_out', img)
    if not helper.isshow and img is not None:
        cv2.imwrite('out_f/' + fileprefix + "_meter_out.jpg", img)

    cv2.waitKey(0)

    grid_1 = 0
    if maxroi is not None:
        for param in improps['guessgridparams']:
            x, y, w, h = maxroi
            roi_area = src[y:y + h, x:x + w].copy()
            param['localthresholvalue']=improps['the_tag_mean']
            param['threshold_plus']=improps['the_tag_std']
            param['fileprefix']=improps['fileprefix']
            param['tagposition']=improps['tagposition']
            th1_level, roi_area, edge_level = processRoi(roi_area,param)
            # get level value
            roi_area, level_list = getlevel(roi_area, edge_level,param)
            print('level_list',level_list)
            grid=guessgrid(maxroi, level_list,param)
            grid_1=max(grid,grid_1)
            cv2.waitKey(0)
            if  (len(level_list)==0 or grid/10<maxroi[3]/improps['tagposition'][3]*0.7 \
                or level_list[0][2]*level_list[0][3]>2000 ):
                continue
            else:
                break

    grid=grid_1

    if grid > 10:
        cv2.putText(origin, 'Result:'+str(grid) + ' grids.', (improps['tagposition'][0]+60, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(origin, (improps['left'],improps['top'] ), (improps['right'], improps['bottom']), (0, 255, 0), 2)
        print(improps['fileprefix']+'刻度值:' + str(grid))
        if helper.isshow:
            helper.imshow(improps['fileprefix']+'meter_grid',origin)
        else:
            cv2.imwrite('out_f/' + improps['fileprefix'] + "meter_grid.jpg", origin)
        cv2.waitKey(0)
    else:
        print(fileprefix,"read grid failed！")
        cv2.putText(origin, 'read meter failed', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        if helper.isshow:
            helper.imshow(fileprefix+'meter_grid',origin)
        else:
            if origin is not None:
                cv2.imwrite('out_f/' + improps['fileprefix'] + "meter_grid.jpg", origin)
    return grid, origin

if __name__ == '__main__':
    if improps_init['fileprefix'] != '':
        start = datetime.datetime.now()
        f = open("out.txt", "w")
        print(improps_init['fileprefix'] + "." + improps_init['filetype'])
        improps_init['path']=improps_init['imagepath'] + '/' + improps_init['fileprefix'] + "." + improps_init['filetype']
        grid, img = readWaterLevel(improps_init)
        print(improps_init['fileprefix'] + "." + improps_init['filetype'] + ":", end="", file=f)
        print(grid, file=f)
        cv2.destroyAllWindows()
        end = datetime.datetime.now()
        print("execution time：", end - start, file=f)
        f.close()
    else:
        start = datetime.datetime.now()
        f = open("out.txt", "w")
        # print(os.listdir(imagepath))
        countall = 0
        count = 0
        countread=0
        helper.isshow=False
        for onefile in os.listdir(improps_init['imagepath']):
            improps=copy.deepcopy(improps_init)
            if os.path.splitext(onefile)[1] == '.jpg':  # 查找图形文件
                countall += 1
                # print(onefile)
                improps['fileprefix'] = os.path.splitext(onefile)[0]
                improps['path']=improps['imagepath'] + '/' + improps['fileprefix'] + "." + improps['filetype']
                print(onefile)
                grid, img = readWaterLevel(improps)

                print(onefile + ":", end="", file=f)
                if grid > 20: countread += 1
                if grid > 0: count+=1
                print(grid, file=f)
                cv2.destroyAllWindows()
        end = datetime.datetime.now()
        print("Execution time:", end - start)
        print("Execution time:", end - start, file=f)
        print("sum:", countall)
        print("successfully located:", count)
        print("successfully read：", countread)
        print("sum：", countall, file=f)
        print("successfully located：", count, file=f)
        print("successfully read：", countread, file=f)

        f.close()
        cv2.waitKey(0)
