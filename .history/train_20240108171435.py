import cv2
import numpy as np
import matplotlib.pyplot as plt
#生成训练集
trainData=np.random.randint(0,100,(25,2)).astype(np.float32)
#生成标签
respose=np.random.randint(0,2,(25,1)).astype(np.float32)
#标签为1则画红色
red=trainData[respose.ravel()==1]
plt.scatter(red[:,0],red[:,1],80,'r','^')
#标签为2则画蓝色
blue=trainData[respose.ravel()==0]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')
#生成测试点
newcomer=np.random.randint(0,50,(1,2)).astype(np.float32)
#创建KNN
knn=cv2.ml.KNearest_create()
#训练
knn.train(trainData,cv2.ml.ROW_SAMPLE,respose)
#测试，k=3
ret,results,neighbours,dist=knn.findNearest(newcomer,3)
#按结果画出点
if results == np.array([[1.0]]):
    plt.scatter(newcomer[:,0],newcomer[:,1],80,'r','o')
else:
    plt.scatter(newcomer[:,0],newcomer[:,1],80,'b','o')
plt.show()
# to do
