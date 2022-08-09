import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from easycm import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler


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


#-------------------------------CNN MODEL-----------------------------------------
#Convolutional neural network (Evrişimsel Siinir Ağları)

model = tf.keras.Sequential()
#Kerasta ki API uygulamasına tf.keras denir. 

#modelin katmanlarını oluşturmak
#modele katman eklemek için add() fonksiyonu kullanılır.
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28,28,3)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))
#10 tane sınıfımız olduğu için 10 kullandım.

model.summary()

#modeli derleyelim

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    steps_per_epoch=25,
    verbose=2,
)
model.fit_generator(train_generator, 
                    steps_per_epoch=100,
                    epochs =2, 
                    validation_data=validation_generator, 
                    validation_steps = 50,
                    verbose=2)



#grafikleri çizdirmek
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = 10 
#Accuracy grafiği
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()
plt.show()
#Loss grafiği
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#confusion_matrix
plot_confusion_matrix(validation_generator, train_generator)

#prediction func

scaler=MinMaxScaler(feature_range=(0,1))
scaler.fit(validation_generator)
print("a")
predict = model.predict(validation_generator)
print("b")
predict = scaler.inverse_transform(predict)
print("c")
print("prediction shape:", predict.shape)
