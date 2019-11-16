#from skimage.transform import rescale
from skimage.transform import resize
from skimage import data, io
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
import codecs, json 
import numpy as np

class ImageProcessing:
    def ImageConverter(fromFile, toFile, width = 28, height = 28):
        #Convert Original image to grayscale
        image_gray = io.imread(fromFile, as_gray=True)

        img_resized = resize(image_gray, (width, height))
        # convert image to numpy array
        data = asarray(img_resized)
        #Convert data to list
        dataList = data.tolist()
        #file path for json file
        file_path = toFile 
        # this saves the data in .json format
        json.dump(c, codecs.open(file_path, 'w', encoding='utf-8'), 
                separators=(',', ':'), sort_keys=True, indent=4) 

    def PlotImage(oriImage, proImage):
        #plot images
        plt.subplot(121), io.imshow(oriImage)
        plt.title('Original Image')
        plt.subplot(122), io.imshow(proImage)
        plt.title('Processed Image')
        plt.show()
