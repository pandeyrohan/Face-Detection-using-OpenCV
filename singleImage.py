# importing the open cv  library
import cv2 as cv
#loading the dataset
trainedData = cv.CascadeClassifier('./training set/haarcascade_frontalface.xml')
#choose the image
img  = cv.imread('./sample/photo.jpg')
#conversion to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#detect faces
facecor = trainedData.detectMultiScale(gray)
x,y,w,h = facecor[0]
cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
#display the image
cv.imshow('img',img)
#pause execution until any key is pressed
cv.waitKey()
print("End Of The Program")

