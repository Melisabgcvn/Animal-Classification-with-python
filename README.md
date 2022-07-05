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

labels=inferred denirse dizin altında her biri bir sınıfa ait görüntüleri içeren alt dizinler olmalı demektir. Bizim örneğimizde de var.

class_names=labels=inferred olarak seçildiğinde kullanılabilir. Sınıf isimlerini gösterir.

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


