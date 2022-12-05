import cv2 as cv
from random import randrange as r
#load the training set
trainData = cv.CascadeClassifier('./training set/haarcascade_frontalface.xml')
#open the webcam
cam = cv.VideoCapture(0)

while True:
    success,img  = cam.read()
    #convert the image to gray scale
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #detect faces
    facecor = trainData.detectMultiScale(gray)
    for x,y,w,h in facecor:
        cv.rectangle(img,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)
    #display the image
    cv.imshow('img',img)
    #pause execution until any key is pressed
    key = cv.waitKey(1)
    if(key == 81 or key == 113): 
        break
cam.release()
print("End Of The Program")
