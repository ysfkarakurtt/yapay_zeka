# YAPAY ZEKA NEDİR?
![](https://shiftdelete.net/wp-content/uploads/2021/11/Yapay-zeka-destekli-insansiz-hava-araclari0.jpg)

* Yapay Zeka(İngilizce *Artificial intelligence*, *AI* ) bir sürü tanımı vardır ama en basit olarak makinelerin insanlar gibi düşünmesini sağlamak amacıyla ortaya çıkan bir alandır.

* Yapay zekâ çalışmaları sıklıkla insanın düşünme yöntemlerini taklit eden yapay yöntemler geliştirmeye yöneliktir."Yapay zekâ" kavramının geçmişi modern bilgisayar bilimi kadar eskidir. Fikir babası, "Makineler düşünebilir mi?" sorunsalını ortaya atarak makine zekâsını tartışmaya açan Alan Mathison Turingdir. 
 
 ![Resimdeki  Alan Mathison Turing](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Alan_Turing_Aged_16.jpg/330px-Alan_Turing_Aged_16.jpg)
* [*Hayatını incelemek istiyorsanız tıklayınız*](https://tr.wikipedia.org/wiki/Alan_Turing)

* “Yapay zeka” ismi ise resmi olarak ilk kez 1956 yılında bir konferansta John McCarty tarafından ortaya atılmıştır ve temelleri oluşturulmuştur. Yani McCarty’nin yapay zeka kavramının isim babası olduğunu söyleyebiliriz. 1959 yılında ise Massachusetts Teknoloji Enstitüsü’nde John McCarty ve Marvin Minsky tarafından ilk yapay zeka laboratuvarı kurulmuştur.
  ![John McCarty](https://www.webtekno.com/images/editor/default/0002/73/87074abc0db40b8f033018040717c0faef7e9809.jpeg)

* Genel olarak yapay zeka  zekanın anlamı ve önemi ,en önemli yapay zeka teknolojileri,dünyanın önde gelen teknoloji şirketlerinden lider görüşlerini ve daha fazlasını ögrenmek için aşağıda bıraktığım linkten *Global AI Hub* tarafından düzenlenen ***Yapay Zekaya İlk Adım*** 
kursunu bitirmenizi öneririm.Yaklaşık 3-4 saatinizi alacak bu kurs size yapay zeka hakkında bir fikir edinmenizi sağlayacaktır ve kurs sonunda sertifika kazanabiliyorsunuz.
[*Yapay zekaya ilk adım kursuna gitmek için tıklayınız.*](https://globalaihub.com/courses/yapay-zekaya-ilk-adim/)

# Makine Öğrenmesi nedir?
![](https://miro.medium.com/max/828/0*BtgG25EaA94fM41O)

* Makine öğrenmesi, kod yazmanıza gerek kalmadan, size belirli bir veri kümesi hakkında ilginç şeyler söyleyebilen genel (generic) algoritmalar oluşturma fikridir. Kod yazmak yerine bu genel algoritmayı veri ile beslersiniz ve bu şekilde algoritma, bu veriye dayanarak kendi mantığını oluşturur.

* Örneğin, sınıflandırma algoritması bu algoritma çeşitlerinden biridir. Sınıflandırma algoritması verileri farklı gruplara ayırır. Aynı sınıflandırma algoritması elle yazılmış sayıları tanımlamada kullanılabileceği gibi, bir satır kod değiştirmeden, epostaları spam veya spam-değil olarak iki gruba ayırmada da kullanılabilir. Algoritma aynı algoritmadır ama farklı veri ile eğitildiği için farklı bir sınıflandırma mantığı oluşturur.

![](https://miro.medium.com/max/1400/1*k8o6ok1h-Cpc3FnF4jn9mQ.webp)


***Bu makine öğrenmesi algoritması farklı sınıflandırma problemleri için kullanılabilen bir kara kutudur.***


* “Makine öğrenmesi” terimi bu tip farklı genel algoritmaları tek bir çatı altında toplayan üst bir kavramdır.

* Makine öğrenmesi algoritmaları ikiye ayrılır:**Gözetimli** ve **Gözetimsiz** öğrenme.
![](https://miro.medium.com/max/720/1*UBeCbdU1UtfVZsheIJqHoQ.webp)


## GÖZETİMLİ ÖĞRENME:
* Gözetimli öğrenme (Supervised learning): Bu öğrenme türünde model eğitilirken her bir veri noktasındaki doğru sonuç açık bir şekilde etiketlenir. Bu, veri okunurken zaten öğrenme algoritmasına cevap verildiği anlamına gelir. Cevap bulmaktan ziyade, veriler arasındaki ilişkinin bulunması amaçlanır.

* Gözetimli öğrenmeyi daha iyi anlayabilmek için bir de örnek üzerinden anlamaya çalışalım 

--> Diyelim ki siz bir emlakçısınız. İşleriniz büyüyor ve size yardım etmesi için birçok stajyer işe aldınız. Ama bir problem var siz bir eve baktığınızda evin değeri hakkında iyi bir tahminde bulunabiliyorsunuz ama stajyerlerinizin tecrübesi olmadığından nasıl değer biçmeleri gerektiğini bilmiyorlar.

-->Stajyerlere yardım etmek amacıyla (ve kendinizi tatil için boşa çıkarmak maksadıyla), sizin bölgenizdeki ev fiyatlarını genişlik, muhit ve benzer evlerin kaça satıldığı vb. gibi özelliklere göre hesaplayan basit bir uygulama yazmaya karar verdiniz.

-->Bu yüzden son üç ayda şehirde satılan tüm evlerin fiyatlarını kaydettiniz. Satılan her evin oda sayısı, genişliği, muhiti vb. gibi detaylı özelliklerini not aldınız. Ama en önemlisi nihai satış fiyatını da kaydettiniz:
![](https://miro.medium.com/max/828/1*nywb702xDKrWBbnOW5_0MA.webp)

***Bu bizim “eğitim verimiz”***

-->Bu eğitim verisini kullanarak, bizim bölgemizdeki diğer tüm evlerin satış fiyatlarını tahmin eden bir program yazmak istiyoruz:
![](https://miro.medium.com/max/828/1*usmXLZiKVyk-bhC-F7wBvA.webp)

***Eğitim verisini kullanarak diğer evlerin fiyatlarını tahmin etmek istiyoruz.***

-->İşte bu yöntem gözetimli öğrenmedir. Her bir evin kaça satıldığını biliyorsunuz, yani problemin cevabını biliyorsunuz ve oradan yola çıkarak geriye doğru bir mantık oluşturmaya çalışıyorsunuz.

-->Uygulamanızı geliştirmek için, her bir eve ait eğitim verisiyle makine öğrenmesi algoritmanızı besliyorsunuz. Algoritma bu sayılar arasındaki ilişkiyi oluşturmak için nasıl bir matematik gerektiğini çözmeye çalışıyor.

-->Bu çeşit problemler aslında şuna benziyor: Matematik testinde elinizde cevaplar var ama aradaki aritmetik semboller silinmiş.

![](https://miro.medium.com/max/828/1*RyvJHk2bgHAvtbFt8z6ZbA.webp)
-->Buradan yola çıkarak testte ne çeşit matematiksel problemler olduğunu çözebilir misiniz? Sağdaki her bir cevaba ulaşmak için soldaki sayıları kullanarak “bir şeyler yapmanız” gerektiğini biliyorsunuz.

--> *Gözetimli öğrenmede, bu aradaki ilişkiyi kendiniz çözmek yerine, bu işi bilgisayara bırakıyorsunuz. Ve belirli bir problem grubunu çözmek için gereken matematiksel ilişkiyi bildiğiniz anda, artık o tipteki tüm problemlere cevap verebiliyorsunuz.*

### En çok kullanılan gözetimli öğrenme algoritmaları:
(***Algoritmaların üstüne basarak detaylı bilgi alabilirsiniz***)
* [En Yakın Komşuluk → k-Nearest Neighbors (KNN)](https://bilgisayarkavramlari.com/2008/11/17/knn-k-nearest-neighborhood-en-yakin-k-komsu/)

* [Yapay Sinir Ağları → Artificial Neural Network (ANN)](https://bilgisayarkavramlari.com/2008/10/02/yapay-sinir-aglari-artificial-neural-networks/)

--> [Yapay sinir ağları ile ilgili güzel bir makale için tıklayınız](https://medium.com/@k.ulgen90/makine-%C3%B6%C4%9Frenimi-b%C3%B6l%C3%BCm-3-4b160df1f4c8)

* [Destek Vektör Makinaları → Support Vector Machine (SVM)](https://medium.com/@k.ulgen90/makine-%C3%B6%C4%9Frenimi-b%C3%B6l%C3%BCm-4-destek-vekt%C3%B6r-makineleri-2f8010824054) 


* [Karar Ağaçları → Decision Trees (DTs)](https://medium.com/@k.ulgen90/makine-%C3%B6%C4%9Frenimi-b%C3%B6l%C3%BCm-5-karar-a%C4%9Fa%C3%A7lar%C4%B1-c90bd7593010)
* [Doğrusal Regresyon → Linear Regression](https://medium.com/kodcular/makine-%C3%B6%C4%9Frenimi-b%C3%B6l%C3%BCm-6-regresyon-3d837236eb6b)
  
  ---> Makine öğrenime giriş için,makine öğrenim türlerini, linear regressionun bahsedildiği **Regresyon Least Square Error(En Küçük Kare Hatalari)** anlattığı ve en son videoada java ile **Least Square Error** programını kodladığı kısa güzel bir playlist için [***Tıklayınız***](https://www.youtube.com/watch?v=FKoFVzTMRog&list=PLstEgQdEnMSaU-tyKVrpuFVveVeqWfg0b)



* [Lojistik Regresyon → Logistic Regression](https://medium.com/@k.ulgen90/lojistik-regresyon-makine-%C3%B6%C4%9Frenimi-b%C3%B6l%C3%BCm-7-c6bc685a4084)


## GÖZETİMSİZ ÖĞRENME:
Gözetimsiz Öğrenme (Unsupervised Learning):

Gözetimsiz öğrenme yönteminde ise label bilgisi yoktur. Veri setindeki bileşenler temel alınarak saklı ilişkilerin veya grupların ortaya çıkarılması amaçlanmaktadır. Örneğin; 10 tane bilgisayar alacaksınız ve bütçeniz sınırlı. Erken bir saatte teknoloji mağazasına gittiğinizi ve bilgisayar fiyatlarıyla ilgili herhangi bir bilgi alamadığınız farz edin. Fakat elinizde her bir bilgisayarın detaylı özellikleri mevcut. Bu durumda ne yapardınız? Gözetimsiz öğrenme yöntemiyle, donanımsal ve yazılımsal özellikler dikkate alınarak fiyat tahmini (label) yapmaya başlıyorsunuz :)

-->Yine baştaki emlakçı örneğine geri dönelim. Her evin satış fiyatını bilmeseydiniz ne olurdu? Tüm bildiğiniz evin genişliği, yeri vb. gibi bilgiler olsa bile, görünen o ki hala işe yarar hesaplamalar yapabilirsiniz. Buna da **gözetimsiz öğrenme** deniyor.
![](https://miro.medium.com/max/640/1*GaoWKVcaGOez2NVtzGKpyw.webp)

***Bilinmeyen bir sayıyı (fiyat gibi) tahmin etmeye çalışmasanız bile, makine öğrenmesiyle ilginç şeyler yapabilirsiniz.***

-->Bu yöntem şuna benziyor: Birisi size bir kağıtta sayı listesi veriyor ve şunu diyor “bu sayıların ne ifade ettiğini bilmiyorum ama belki sen burada bir düzen veya grup gibi birşey bulabilirsin.



-->Öyleyse bu veriyle ne yapılabilir? Yeni başlayanlar, veriniz içindeki farklı piyasa segmentlerini tanımlayan bir algoritma kullanabilirsiniz. Okulun yakınındaki evleri alanların küçük ama çok sayıda odası olan evleri tercih ettiğini buna karşı kenar mahallelerde ev alanların 3 odalı ve büyük evleri tercih ettigini keşfedebilirsiniz. Bunun gibi farklı tip müşteriler hakkında bilgi sahibi olmanız pazarlama çalışmalarınızı yönlendirmeniz için size yardımcı olabilir.

-->Yapabileceğiniz diğer güzel bir şey ise normalin dışında kalan, diğerlerinden çok farklı evleri otomatik olarak tespit edebilmektir. Bu evler sıra dışı devasa malikaneler olabilir ve siz en iyi satıcılarınızı bu bölgelere yönlendirebilirsiniz, çünkü bu evlerin komisyonları daha yüksek olacaktır.

### En Çok Kullanılan Gözetimsiz Öğrenme:
* [Kümeleme → Clustering](https://medium.com/@ekrem.hatipoglu/machine-learning-clustering-k%C3%BCmeleme-k-means-algorithm-part-13-be33aeef4fc8)

* [Birliktelik Kuralları → Association Rules](https://www.veribilimiokulu.com/associationrulesanalysis/)

* [Temel Bileşen Analizi → Principal Component Analysis (PCA)](https://medium.com/@gulcanogundur/pca-principal-component-analysis-temel-bile%C5%9Fenler-analizi-bf9098751c62#:~:text=T%C3%BCrk%C3%A7esi%20%E2%80%9CTemel%20Bile%C5%9Fenler%20Analizi%E2%80%9D%20olan,indirgemeyi%20sa%C4%9Flamak%20olan%20bir%20tekniktir.)

------------------------
#### Yapay zekanın geçmişten günümüze hayatımızdaki yerinden örneklerle bahseden güzel bir TEDx konuşması [***Yapay Zekanın Girdabı Cem Say TEDx konuşmasını izlemek için tıklayınız.***](https://www.youtube.com/watch?v=HNVA4nLzkpQ)
-------------
#### Kitap okumayı sevenler için makine öğrenmesinin anlatıldığı başucu olan kitaplardan bir tanesi [kitaba gitmek için tıklayınız](https://www.amazon.com.tr/Scikit-Learn-TensorFlow-Uygulamal%C4%B1-Geli%C5%9Ftirmek-Konseptler/dp/6050624828/ref=asc_df_6050624828/?tag=trshpngglede-21&linkCode=df0&hvadid=540942491897&hvpos=&hvnetw=g&hvrand=4983035058394969732&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1012782&hvtargid=pla-1460676299885&psc=1) (Eğer ingilizce seviyeniz iyi ise orijinal halinden okumanızı öneririm)

--------------------------


### Kitap okumayı sevenler için derin öğrenmenin anlatıldığı  en iyi kitaplardan bir tanesi [kitaba gitmek için tıklayınız](https://www.dr.com.tr/Kitap/Python-ile-Derin-Ogrenme/Francois-Chollet/Egitim-Basvuru/Egitim/urunno=0001874770001)
---------

 ### ---->Yapay zeka ve makine öğrenmesinde daha ileri gitmek isteyenler için çok iyi bilinen ünlü Makine Öğrenimi öğretmeni [**Andrew Ng**](https://www.coursera.org/instructor/andrewng) verdiği kursu da kullanılabilirsiniz. Bu ders alacaklarınızın içinde ki en kaliteli derslerden biri olacaktır. Ücretsiz coursera.org üzerinden ulaşabilirsiniz.Kursa gitmek için [**tıklayınız**](https://www.coursera.org/learn/ai-for-everyone)
