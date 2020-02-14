from PIL import Image, ImageDraw
from PIL import ImageFilter
import os
import numpy as np
import string
import random


dir_path = os.path.dirname(os.path.realpath(__file__))

def createIm(text):
    #Turns input text to image matrix. Also returns a blurred version of the image
    img = Image.new('L', (50, 10), color = (0))
    d = ImageDraw.Draw(img)
    d.text((8,0), text, fill=(255))
    blurImg = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    return (np.array(img),np.array(blurImg))

def randomStr(length=6):
    #Generate random 6 uppercase letter word
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(length))

def Iter(trainDataSize=100000):
    #Generate pair of numpy matrices depicting random text images and blurred text images
    imData = np.empty([trainDataSize,10,50])
    imBlurData = np.empty([trainDataSize,10,50])
    for i in range(0,trainDataSize):
        im,blurIm = createIm(randomStr())
        imData[i,:,:] = im
        imBlurData[i,:,:] = blurIm
    return imData,imBlurData
if __name__ == "__main__":
    (xData,yData) = Iter()
    #Save generated training data
    np.save('x_train',xData)
    np.save('y_train',yData)
    #Save and generate test data
    (xData,yData) = Iter(1000)
    np.save('x_test',xData)
    np.save('y_test',yData)
