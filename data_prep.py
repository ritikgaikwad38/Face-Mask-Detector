#!/usr/bin/env python
# coding: utf-8

from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import cv2 
from tqdm import tqdm

# python -m pip install --upgrade pip
# pip install opencv-python

#download link: https://drive.google.com/u/0/uc?id=1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp&export=download

datasetPath = "dataset"
maskFacePath = datasetPath+"/AFDB_face_dataset"
noMaskFacePath = datasetPath+"/AFDB_masked_face_dataset"
outputDataFile = "data.csv"

data = []
print("Getting list dirs: ", maskFacePath)
for subject_id in listdir(maskFacePath):
    for image_id in listdir(maskFacePath+"/"+subject_id):
        data.append([maskFacePath+"/"+subject_id+"/"+image_id, 0])

print("Getting list dirs: ", noMaskFacePath)
for subject_id in listdir(noMaskFacePath):
    for image_id in listdir(noMaskFacePath+"/"+subject_id):
        data.append([noMaskFacePath+"/"+subject_id+"/"+image_id, 1])

print("building pandas DataFrame")
df = pd.DataFrame(data, columns = ['image_path', 'hasMask']) 
df = shuffle(df).reset_index(drop=True)
df.head()
print(df.shape)

print("Saving pandas dataframe to csv: ", outputDataFile)
df.to_csv(outputDataFile, sep=',', encoding='utf-8', index=False)

img_size = 100
img_raw_data = []
target=[]
i = 0
raw = df.values
print("Preparing images for ML models: ")
for i in tqdm(range(len(raw))):
    image = cv2.imread(raw[i, 0])
    try:
        #resizing the gray scale into 100x100,
        #since we need a fixed common size for all
        #the images in the dataset
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized=cv2.resize(gray,(img_size,img_size))
        img_raw_data.append(resized)
        target.append(raw[i, 1])
    except Exception as e:
    	print('Exception:', e)

#normalize
data=np.array(img_raw_data)/255.0
#restructure data for cnn
data=np.reshape(data,(data.shape[0],img_size,img_size,1))

target=np.array(target)

print("Saving pre-processed images for ML models")
np.save('imageData', data)
np.save('target', target)




