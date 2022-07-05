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
path = "raw-img" #bilgisayarımda kullanacağı dosyanın yolunu path adlı variable da tutucam.
classes = os.listdir(path) #dosya içindeki sınıfların isimlerini görmek için

#veri yükleme
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "raw-img", batch_size=64, image_size=(224, 224))
#batch_size aşağıda açıklanmıştır.

#İkinci aşama olarak yüklenen veriler train, validation verilerine bölünmeli.

train_datagen = ImageDataGenerator(
        validation_split=0.25 #verilerimizi hangi oranda böleceğimizi söylüyoruz.
          )

train_generator = train_datagen.flow_from_directory(
        path,
        subset='training',
        #veri büyütme :
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
        
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

# =============================================================================  
#Batch :
#     Batch işleminde, veri seti batch değeri olarak belirlenen değere göre parçalara ayrılmakta ve her iterasyonda modelin eğitimi bu parça üzerinden yapılmaktadır.
#     Bununla birlikte bazı durumlarda veri kendi içinde gruplanmış olabilmektedir.
#     Bu durum veri seti içinde korelasyon oluşturacak; bu veri setinden seçilecek test setin de yüksek başarım vermesini sağlayacak böylece ezberleme (“overfitting”) olacaktır. 
#     Bunu önlemek için eğitim başlamadan veri seti parçalara ayrılmadan önce veriseti karıştırılmalıdır (shuffle). Batch seçiminde verilerin rastgele seçilmesi önemlidir.
#     Batch size küçük olması iyileştirme (reguralization) etkisi yaratmaktadır. Modele veri büyük gruplar halinde verildiğinde ezberleme daha fazla oluyor.
#     Batch boyutunun diğer bir kıstası da bellek boyutudur. Eğer küçük belleğe sahip ortamda çalışıyorsanız, batch büyük tutmakta zorlanabilirsiniz. Bu nedenle modeli tasarlarken öncesinde kullanabileceğiniz maksimum batch değeri hesaplamak verimli olacaktır.
#     
# =============================================================================
    
    
# =============================================================================
# Found 26179 files belonging to 10 classes.
# Found 19638 images belonging to 10 classes. #train verileri
# Found 6541 images belonging to 10 classes.  #validation verileri
# =============================================================================

