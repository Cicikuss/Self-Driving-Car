# Behavioral Cloning

Behavioral Cloning (BC), uzman bir karar vericiden (expert) doğrudan örnek öğrenerek davranış politikası çıkarma yaklaşımıdır. Geleneksel reinforcement learning'de olduğu gibi deneme-yanılma ile öğrenme yerine, BC problemi supervised learning olarak ele alınır. Yani model, gözlem ile buna karşılık gelen uzman aksiyonu arasındaki eşleştirmeyi öğrenmeye çalışır.

Bu yöntem, politika öğrenmek için oldukça basit ve uygulanabilir bir yaklaşımdır. Ancak genellikle genelleme kapasitesi sınırlıdır ve daha önce görülmeyen durumlarda ya da küçük hatalarda kendini kurtarmada zayıf kalabilir. Bu yüzden DAgger, GAIL gibi alternatif yöntemler de geliştirilmiştir; ancak bu yazıda bunlara değinmeyeceğim.

---

## Nasıl Çalışır?

### 1) Veri Toplama
Uzmanın deneyimlerinden elde edilen gözlem ve aksiyon çiftleri toplanır. Bu veriler, modelin öğrendiği örneklerdir. Örneğin, direksiyon açısı ile o anda görüntülenen yol görünümü gibi bir çift oluşturulabilir.

Ayrıca, veri setine augmentation da eklenebilir. Örneğin:
- resmi döndürmek,
- ışık ayarını değiştirmek,
- görseli ötelemek ya da hafifçe bozmaktır.

Bu tür işlemler, modelin daha sağlam ve genelleştirilebilir öğrenmesi için yararlı olabilir.

### 2) Politika Eğitimi
Model, uzman tarafından verilen aksiyonu mümkün olduğunca doğru tahmin etmeye çalışır. Eğitim sürecinde loss fonksiyonu minimize edilir. Benim durumumda action space sürekli olduğu için (continuous action space) ortalama karesel hata (MSE) kullandım.

Yani modelin amacı aşağıdaki gibi özetlenebilir:

- giriş: gözlem / görüntü / durum bilgisi
- çıktı: uzman tarafından verilen uygun aksiyon
- hedef: bu çıktıyı mümkün olduğunca doğru tahmin etmek

### 3) Deployment
Eğitim tamamlandıktan sonra model, bir ortam içinde çalıştırılır. Ben bu yaklaşımı Udacity Self Driving Car simülasyonunda denedim. Bu süreçte model, gerçek zamanlı olarak görüntüden elde ettiği bilgiyi kullanarak sürüş aksiyonunu üretmeye çalışır.

---

## Nerede Kullanılır?

Behavioral Cloning, özellikle uzman davranışlarının örneklenebildiği alanlarda kullanılır. Örnekler:

- otonom sürüş,
- robotik kontrol,
- oyun oynama davranışları,
- uzman rehberliği ile öğrenme gerektiren sistemler.

Öne çıkan örneklerden biri NVIDIA'nın DAVE-2 sistemidir. Bu sistem, sürüş davranışını uzman sürücüden öğrenerek otonom sürüş için kullanmıştır. Benzer şekilde robotikte de uzman hareketlerinin taklit edilmesi için kullanılabilir.

---

## Limitasyonları ve Dezavantajları

Behavioral Cloning en basit ve uygulanabilir yöntemlerden biri olsa da bazı önemli zayıflıkları vardır:

- model yalnızca uzman verileri üzerinde eğitildiği için, deployment sırasında yaptığı küçük hatalar daha önce hiç görmediği durumlara sürükleyebilir;
- bu durum, modelin doğru karar veremediği yeni state'lere girmesine neden olabilir;
- genelleme yeteneği çoğu zaman sınırlıdır;
- uzman davranışını taklit etme kapasitesi iyi olsa da, hata durumlarında kendini toparlama becerisi zayıf olabilir.

Bu yüzden bazı durumlarda DAgger gibi yöntemler daha iyi sonuç verebilir. Bu yöntem, modelin hatalı durumlarla karşılaştıkça yeni veri toplayarak öğrenmesini sağlar. Ben de bu sorunu DAgger ile çözmeye çalıştım.

---

## Avantajları

Behavioral Cloning'in güçlü yönleri şunlardır:

- uygulanması nispeten basittir;
- önceden kaydedilmiş veri seti ile eğitilebilir;
- simülasyon ortamında saatlerce deneme-yanılma beklemek gerekmez;
- reward hacking gibi sorunlar, reinforcement learning tabanlı yöntemlerde daha belirgin olsa da BC'de bu durum daha az görülür;
- uzman davranışını doğrudan taklit etmesi nedeniyle hızlı prototip geliştirme için uygundur.

---

## Sonuç

Behavioral Cloning, uzman davranışlarını taklit etmek için güçlü ve pratik bir yöntemdir. Özellikle veri mevcutsa ve hedef davranış belli bir uzman tarafından tanımlanabiliyorsa etkili sonuçlar verebilir. Ancak genelleme ve hata durumları açısından bazı sınırlamaları vardır. Bu nedenle, basitlik avantajı nedeniyle çok yaygın olsa da, daha zorlu senaryolarda DAgger gibi yöntemler daha uygun seçenekler olabilir.
