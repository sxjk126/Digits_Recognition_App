from skimage.transform import rescale
from skimage.transform import resize
from skimage import data, io
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
import codecs, json 
import numpy as np
image_gray = io.imread(r'D:\Repos\Number_Two.jpg', as_gray=True)
img_resized = resize(image_gray, (28,28))
data = asarray(img_resized)
b = np.array(data)
a = b.flatten()
c = a.tolist()

file_path = r"D:\Repos\NumTwo.json" 
#### this saves the array in .json format
json.dump(c, codecs.open(file_path, 'w', encoding='utf-8'), 
          separators=(',', ':'), sort_keys=True, indent=4) 
