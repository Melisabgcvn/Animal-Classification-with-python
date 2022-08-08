# Animal-Classification-with-python

Bilgisayarda dosya görünümü :


<img width="383" alt="Ekran Resmi 2022-07-05 15 19 33" src="https://user-images.githubusercontent.com/96236352/177325779-7ae86233-3145-491a-a4f3-714df2421fcd.png">


<img width="868" alt="Ekran Resmi 2022-07-05 15 19 44" src="https://user-images.githubusercontent.com/96236352/177325812-4682bbd3-56ef-4037-a034-467451241725.png">

#-----------------------------

!! Kullanılan bazı sınıflar, fonksiyonlar ve özellikleri:

##ImageDataGenerator Sınıfı##

Mevcut verilerin üzerinde rasgele döndürme, kesme, kaydırma gibi işlemler yaoarak eğitim verilerini arttırmaya "Veri büyütme" denir.
Gerçek zamanlı veri büyütme ile tensör görüntü verisi yığınları oluşturur.Veriler gruplar halinde döngüye alınır.

##tf.keras.preprocessing.image_dataset_from_directory##

Veri kümemizi bir dizinden direkt olarak okutabilmemizi sağlar. Bu fonksiyon jpeg, png, bmp, gif formatlarını destekler.

directory=verinin yer aldığı dizin

labels = inferred denirse dizin altında her biri bir sınıfa ait görüntüleri içeren alt dizinler olmalı demektir. Bizim örneğimizde de var.

class_names = "labels=inferred" olarak seçildiğinde kullanılabilir. Sınıf isimlerini gösterir.

batch_size = yığın büyüklüğüdür ve default değeri 32'dir.

label_mode=categorical etiketlerin kategorik bir vektör olarak kodlandığı anlamına gelir. float32 tipli tensörler oluşur.

color_mode=rgb varsayılan değeridir. Görüntülerin 1, 3, 4 kanala dönüştürülüp dönüştürülemeyeceğini gösterir.

shuffle=verilerin karıştırılımasını sağlar. Böylecek overfitting engellenmeye çalışılır.

seed argümanı rasgele değer alır. Amacı shuffle için değer verir.

##flow_from_directory##

Bir ImageDataGenerator metodudur. ImageDataGenerator görüntüler için bir üreticidir.Gerçek zamanlı veri çeşitlendirme (real-time data augmentation) yaparak görüntü verilerini yığınlar olarak oluşturur.
Veri çeşitlendirme, görüntü sınıflandırma, nesne algılama, görüntü bölütleme gibi pek çok yöntemle veri arttırma için kullanılır.

<img width="422" alt="Ekran Resmi 2022-07-05 16 40 39" src="https://user-images.githubusercontent.com/96236352/177341554-14630867-f630-46e5-b92c-d67e20105653.png">

directory=verinin yer aldığı dizin

target_size: Tamsayılardan oluşan bir demettir. Default değeri (256,256)'dır. Çok veri olduğundan bellekte fazla alan kaplamamak adına ben 180,180 ayarladım.

save_to_dir:İsteğe bağlı oluşturulan çeşitlendirilmiş görüntüleri kaydetmek için bir dizin belirtmeye olanak tanır. save_prefix kaydedilen yeni görsellerin isimlerini tutar.save_format kaydedilen görüntülerin formatını tutar.
Çok veri olduğu için ben kaydetmicem.

follow_links:sınıf alt dizinleri içindeki ağların takip edilmeyeceğini belirtir.

subset:ImageDataGenerator snıfında validation_split argümanı ayarlandıysa training ve validation kümeleri oluşturmaya yarar.

interpolation: Yüklenen görüntünün boyutu, target_size argümanıyla tanımlanan hedef boyutundan farklı ise, görüntüyü yeniden örneklemek için kullanılacak interpolasyon yöntemidir. Default değeri nearest'dir.


Artık verileri train ve validation olarak 2 parçaya ayırdık. 

Çıktı resmi:

<img width="382" alt="Ekran Resmi 2022-07-05 16 48 19" src="https://user-images.githubusercontent.com/96236352/177343053-d8c4ae44-3c15-4eec-82ae-e49af277d394.png">


#-------CNN-------#

Bir modelin yaşam döngüsü vardır.

#-----------------------#

1.Model tanımlanır. #model = tf.keras.Sequential()

2.Model derlenir. 

3.Model fit edilir.

4.Model değerlendirilir.

5.Tahminlerde bulunulur.

#------------------------#

2.Modeli Derlemek/

OPTİMİZASYON fonksiyonu seçimi:

Amaç cnn'deki hatayı minimize etmektir. En sık kullanılan optimizasyon yöntemleri:

1.SGD : tüm gradient’ler yerine rastgele olmak üzere bir kısım gradient’le ağırlıkları güncellemektedir.

2.Momentum : SGD’de optimum nokta aranırken çok fazla salınım olmaktadır. Bu salınımları azaltmak ve dolaysıyla hedefe gitme hızını arttırmak için momentum yöntemi önerilmektedir. Bu yöntemde mevcut gradient’ler yerine momentumlu gradient kullanılmaktadır.

3.Adagrad : SGD ve Momentum yöntemlerindeki sabit öğrenme katsayısı problemini ortadan kaldırmak için önerilmiştir. Gradientlerin karelerini alarak hesaplama yapar.

4.RMSProp :Adagrad’da olduğu gibi sabit öğrenme katsayısı problemini çözmek için önerilmiştir. Aralarındaki fark ise,adagrad yöntemindeki gradientlerin karelerini almak yerine momentumlu gradientlerin karelerini almaktadır.

5.Adadelta : Adadelta yönteminde, adagrad ve RMSProp yöntemlerinden farklı olarak öğrenme katsayısı seçme zorunluluğu yoktur. Öğrenme katsayısı yerine, geçerli ağırlıklar ile güncellenen ağırlıklar arasındaki farkı ifade eden delta değerlerinin karelerinin momentumlu
toplamları kullanılmaktadır.

6.Adam : Rmsprop ve momentum yöntemlerinin avantajlı yönlerinin birleştirilmesi ile önerilen gradient descent algoritmasıdır.



KAYIP fonksiyonu seçimi:

Kayıp fonksiyonları seçenekleri arasında :

1.categorical_crossentropy : Etiketler ve tahminler arasındaki çapraz entropi kaybını hesaplar. Yani iki veya daha fazla etiket sınıfı olduğunda kullanılmaya uygundur. Bizim veri setimizde 10 sınıf olduğundan dolayı bunu kullanmak en uygunu gibi duruyor.

2.binary_crossentropy : 2 sınıftan oluian problemlerden birine 1 diğerine 0 diyerek işaretler. (kedi=1 ve köpek=0) gibi. Ama bizim veri setimiz için uygun değildir.

3.MeanSquaredError : Optimizasyon algoritması olarak SGD(stokastik gradyan düşümü) kullanılırken hata fonksiyonu olarak kullanılır. Regresyon hesaplamalarında tercih edilir.


AKTİVASYON Fonksiyonları:

1.Step Fonksiyonu: Bir eşik değeri alarak ikili bir sınıflandırma çıktısı (0 yada 1) üretir.

2.Sigmoid Fonksiyonu: En yaygın kullanılan aktivasyon fonksiyonlarından birisidir, [0,1] aralığında çıktı üretir.

3.Tanh Fonksiyonu: [-1,1] aralığında çıktı üreten doğrusal olmayan bir fonksiyondur.

4.ReLU Fonksiyonu: Doğrultulmuş lineer birim (rectified linear unit- RELU) doğrusal olmayan bir fonksiyondur. ReLU fonksiyonu negatif girdiler için 0 değerini alırken, x pozitif girdiler için x değerini almaktadır. Genel olarak iyi sonuç verir.

5.Softmax Fonksiyonu: Çoklu sınıflandırma problemleri için kullanılan bu fonksiyon, verilen her bir girdinin bir sınıfa ait olma olasılığını gösteren [0,1] arası çıktılar üretmektedir. Bizim problemimiz için uygun gözüküyor.


##LAYERS##


<img width="532" alt="Ekran Resmi 2022-07-07 12 56 54" src="https://user-images.githubusercontent.com/96236352/177746776-a1889b27-382c-42d1-9c63-1094f8b952d3.png">


İlk 2 katmanımız conv2d katmanıdır. Bunlar 2 boyutlu matrisler olarak görülen girdi görüntüleri ile ilgilenir.

İlk katmanımızda 64, ikinci katmanımızda 32 düğüm vardır. Bu sayılar veri kümesinin boyutuna göre değiştirilebilir. Şİmdilik denemeleri bunlar üstünden yapıcam, sonrasında değiştirilebilir.

kernel_size : (çekirdek boyutu) evrişim için filtre matrisinin boyutudur yani 3 çekirdek boyutu, 3x3 filtre matrisine sahip olduğumuz anlamına gelir.

activation : katman için aktivasyon fonksiyonudur. 

input_shape : Keras'ta girdi katmanının kendisi bir katman değil tensördür. İlk gizli katmana gönderidğimiz başlangıç tensörüdür. Bu tensör, eğitim verilerimiz ile aynı şekile sahip olmalıdır. Renk kanalını girdi olarak rgb yani 3 kanal verdiysek bu katmanda da 3 olmalıdır. Girdi pikselleri 50x50 ise burada da öyle olmalıdır. 

Flatten() : Conv2d katmanları ve Dense katmanları arasında flatten yani düzleştirme işlemi yapılır. Düzleştirme katmanlar arasında bir bağlantı görevi görür. 


<img width="308" alt="Ekran Resmi 2022-07-07 13 00 36" src="https://user-images.githubusercontent.com/96236352/177747623-917ec2b5-41c1-4efb-bbe5-95e34e8ee2dd.png">

Dense : Çıktı için kullanacağımız katman türüdür. 


3.Modeli Fit Etmek --------

history = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator,
    steps_per_epoch=25,
    verbose=2,
)









