import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os, sys
import cv2
import shutil
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#diskte bulunann verileri yüklemek için birden fazla yöntem vardır. Bunlar :
    #1 : numpy dizisi oluşturmak
    #2 : tensorflow dataset objeleri oluşturma : belleğe sığmayan, internet üzerinden veya diskten okunacak veriler için kullanılabilecek yüksek performanslı bir veri yapısıdır.
    #3 : Python generator fonskiyonları 
    
    
#Tensorflow ile veri yükleme işlemi yapıcaz.
path = "raw-img"
#classes = os.listdir(path) #sınıfların isimlerini görmek için. Ama biz zaten labels="inferred dediğimiz için otomatik zaten isimleri görebileceğiz.


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "raw-img", 
    labels='inferred', 
    batch_size=32, #tf.data.Dataset nesnesi her seferinde 32 tane görüntü döndürür.
    image_size=(180, 180), #boyut
    color_mode="rgb",
    shuffle=True,
    seed=0
    )

#print(dataset.class_names) 
#labels="inferred" dediğimiz için otomatik olarak class_name özelliğini kullanabildik.

#İkinci aşama olarak yüklenen veriler train, validation verilerine bölünmeli.

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
        path,
        subset='training',
        target_size=(180, 180), #görüntüler boyutlandırılır.
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        follow_links=False,
        interpolation='nearest'
        )

validation_generator = train_datagen.flow_from_directory(
        path, #same as in train generator
        subset='validation', #
        target_size=(224, 224), #giriş resimlerinin boyutudur.
        batch_size=32,
        class_mode="categorical",
        shuffle=True, #Verilmekte olan görüntünün sırasını karıştırmak  için
        seed=42 #Rastgele görüntü büyütme uygulamak ve görüntünün sırasını karıştırmak için 
        
        )

##ÇIKTI##
# =============================================================================
# Found 26179 files belonging to 10 classes.
# Found 19638 images belonging to 10 classes. #train verileri
# Found 6541 images belonging to 10 classes.  #validation verileri
# =============================================================================


#BU for döngüsü ile verilerden bazılarını görebiliriz. Örnek olarak ekledim. Şu an için gerekli değil.
# =============================================================================
# 
# class_names = dataset.class_names
# plt.figure(figsize=(10, 10))
# for images, labels in dataset.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i].numpy().argmax()])
#         plt.axis("off")
# 
# =============================================================================

