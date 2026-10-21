# Self-Driving Car

Bu proje, kamera görüntülerinden direksiyon açısını tahmin eden bir davranışsal
klonlama (behavioral cloning) modeli ile Unity tabanlı bir sürüş simülatöründe
otonom sürüş denemeleri yapmak için hazırlanmıştır.

Model, NVIDIA'nın davranışsal klonlama mimarisine benzer bir evrişimsel sinir
ağı kullanır. Eğitim verisi, sürüş sırasında kaydedilen kamera görüntüleri ve
direksiyon açılarını içerir.

## Özellikler

- Kamera görüntülerinden direksiyon açısı tahmini
- Veri dengeleme ve görüntü artırma (augmentation)
- Eğitim ve doğrulama kaybı grafiklerinin oluşturulması
- Udacity Self Driving Car simülatörü ile Socket.IO üzerinden iletişim
- Model aktivasyonlarının canlı olarak görselleştirilmesi
- Klavye ile uzman sürüş verisi toplamak için DAgger altyapısı

## Proje yapısı

```text
.
├── training/
│   ├── Behavioral Cloning.py  # Model eğitimi
│   └── utils.py               # Veri işleme, augmentation ve model mimarisi
├── dagger.py                  # Uzman sürüş verisi toplama yardımcısı
├── drive.py                   # Simülatöre bağlanıp model ile sürüş
├── visualize.py               # Ara katman aktivasyonlarını gösterme
├── requirements.txt            # Python bağımlılıkları
├── model.keras                # Hazır model ağırlıkları
├── Data/                       # Sürüş verileri
└── Default Windows desktop 64-bit.exe
                                # Udacity Self Driving Car simülatörü (Windows)
```

## Gereksinimler

- Windows
- Python 3.10 veya üzeri
- Udacity Self Driving Car Sim
- Python sanal ortamı kullanılması önerilir

Kurulum:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Veri formatı

Eğitim veri dizini aşağıdaki dosya yapısına sahip olmalıdır:

```text
Data/
└── Tracks/
    ├── driving_log.csv
    └── IMG/
        ├── center_*.jpg
        └── ...
```

`driving_log.csv` sütunları:

```text
center,left,right,steering,throttle,reverse,speed
```

Mevcut eğitim akışı merkez kamera görüntülerini kullanır. Görüntüler kırpılır,
YUV renk uzayına dönüştürülür, bulanıklaştırılır ve `200x66` boyutuna
ölçeklendirilir.

## Modeli eğitme

Proje kök dizininden:

```powershell
python "training\Behavioral Cloning.py" --data-dir "Data\Tracks" --output "model.keras"
```

Yaygın seçenekler:

```powershell
python "training\Behavioral Cloning.py" `
  --data-dir "Data\Tracks" `
  --output "model.keras" `
  --epochs 10 `
  --batch-size 100 `
  --steps-per-epoch 300 `
  --validation-steps 200 `
  --no-plot
```

Eğitim sonunda model, `--output` ile verilen konuma kaydedilir. Grafiklerin
gösterilmesini istemiyorsanız `--no-plot` seçeneğini kullanabilirsiniz.

## Simülatörde sürüş

1. Udacity Self Driving Car simülatörünü başlatın.
2. Model dosyasının `drive.py` içindeki model yolu ile eşleştiğini kontrol edin.
   Kod şu anda `modelc.keras` dosyasını yüklemeyi denemektedir; depodaki hazır
   modelin adı ise `model.keras` olarak verilmiştir.
3. Simülatörde bağlantıyı başlatın.
4. Proje kök dizininden sunucuyu çalıştırın:

```powershell
python drive.py
```

Sunucu varsayılan olarak `4567` portunu dinler. `visualize.py` kullanıldığı
için sürüş sırasında aktivasyon penceresi açılır. Sol ve sağ ok tuşları ile
gösterilen katman değiştirilebilir.

## DAgger ile veri toplama

`dagger.py`, uzman sürüş sırasında görüntüleri, direksiyon açısını, gazı,
geri vitesi ve hızı zaman damgalı bir veri dizinine kaydetmek için
`DAgger` sınıfını sağlar. `A` ve `D` tuşları direksiyon açısını değiştirir;
bırakıldığında açı sıfırlanır.

Her kayıt için görüntü ve `driving_log.csv` üretilecek şekilde tasarlanmıştır.
Bu yardımcı şu anda bağımsız bir veri toplama bileşenidir; simülatör telemetrisi,
model politikası ve yeniden eğitim döngüsüne tam olarak bağlanması sonraki
geliştirmelerin parçasıdır.

## Gelecek geliştirmeler

### DAgger entegrasyonu

DAgger (Dataset Aggregation) ile modelin kendi ürettiği durumlar üzerinde
uzman düzeltmeleri toplanacaktır. Planlanan akış:

1. Mevcut model ile simülatörde sürüş yapma
2. Modelin karar verdiği durumlarda uzmandan düzeltme alma
3. Model ve uzman örneklerini ortak veri kümesinde birleştirme
4. Birleşik veri kümesiyle modeli yeniden eğitme
5. Birden fazla iterasyon sonunda performansı karşılaştırma

### GAIL entegrasyonu

GAIL (Generative Adversarial Imitation Learning), yalnızca etiketli
direksiyon açılarını taklit etmek yerine uzman sürüş davranışının durum-eylem
dağılımını öğrenmek için ileride eklenecektir. Bu aşamada simülatör ortamı,
eylem uzayı, ödül/ayrıştırıcı tasarımı ve eğitim değerlendirme metrikleri
belirlenecektir.

Bu iki yöntem uygulanana kadar mevcut temel yöntem, gözetimli davranışsal
klonlama eğitimidir.

## Notlar

- Eğitim ve sürüşte kullanılan ön işleme adımlarının aynı tutulması gerekir.
- Gerçek sürüşten önce simülatörde düşük hızla test yapılmalıdır.
- Büyük veri kümeleri ve model dosyaları Git'e eklenmeden önce `.gitignore`
  kuralları kontrol edilmelidir.
