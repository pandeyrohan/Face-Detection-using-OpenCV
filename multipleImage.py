#importing the OpenCV
import cv2 as cv
from random import randrange as r
#loading the dataset
trainData = cv.CascadeClassifier('./training set/haarcascade_frontalface.xml')
#loading the image
img   = cv.imread('./sample/group.jpg')
#convert the image to gray scale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#detect faces
facecor = trainData.detectMultiScale(gray)
for x,y,w,h in facecor:
    cv.rectangle(img,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)
#display the image
cv.imshow('img',img)
#pause execution until any key is pressed
cv.waitKey()
print("End Of The Program")

