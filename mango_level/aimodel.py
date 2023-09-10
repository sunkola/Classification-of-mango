#pip install Pillow

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

class AI_model():
    def __init__(self, model_path, class_name):
        self.model = load_model(model_path, compile=False)
        with open(class_name, 'r') as f: #'r'唯讀模式(檔案需存在)，只能從指定檔案讀取資料，並不能夠對這個檔案的內容進行任何寫入或變更
            self.class_names = f.readlines()#with open('文件名', '模式') as 變數名稱:
        
        self.testgenerator = ImageDataGenerator(
            rescale = 1./255,
            preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
        )
    
    def predict(self, img_file):
        image = load_img(img_file, target_size=(180, 180))
        image_array = img_to_array(image)
        image_array = self.testgenerator.standardize(image_array)
        
        data = np.expand_dims(image_array, axis=0)
        
        prediction = self.model.predict(data)
        index = np.argmax(prediction)
        class_name = self.class_names[index].strip()
        confidence_score = prediction[0][index]
        
        return confidence_score, class_name


