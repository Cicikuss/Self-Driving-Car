# Behavioral Cloning for a Self-Driving Car

Bu proje, kamera görüntülerinden direksiyon açısını tahmin eden bir davranışsal
klonlama (behavioral cloning) modeli eğitmek ve bu modeli simülatörde
çalıştırmak için hazırlanmıştır. Kod tabanı, NVIDIA benzeri bir CNN mimarisi,
veri ön işleme adımları, eğitim akışı ve isteğe bağlı aktivasyon
görselleştirmesi içerir.

## Özellikler

- Kamera görüntülerinden direksiyon açısı tahmini
- Eğitim verisi dengeleme ve veri artırma
- Eğitim/validasyon kaybı grafiklerinin oluşturulması
- Socket.IO ile simülatör iletişimi
- Aktivasyon görselleştirme
- DAgger benzeri manuel düzeltme verisi toplanması

## Proje yapısı

```text
.
├── training/
│   ├── behavioral_cloning.py     # Ana eğitim betiği
│   ├── behavioral_cloning_corretion.py
│   └── utils.py                  # Veri yükleme, ön işleme, artırma ve model tanımı
├── dagger.py                     # DAgger veri toplama yardımcı betiği
├── drive.py                      # Simülatörde model ile sürüş
├── visualize.py                  # Aktivasyon görselleştirme
├── requirements.txt              # Python bağımlılıkları
├── README.md                     # Proje dokümantasyonu
├── .gitignore
└── Data/                         # Eğitim verisi klasörü
```

## Gereksinimler

- Python 3.10+
- Sanal ortam önerilir
- Udacity Self-Driving Car simülatörü veya uyumlu bir simülatör
- GPU opsiyoneldir, eğitim hızını artırır

## Kurulum

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Veri formatı

Eğitim verisi aşağıdaki yapıda olmalıdır:

```text
Data/
└── Tracks/
    ├── driving_log.csv
    └── IMG/
        ├── center_*.jpg
        ├── left_*.jpg
        ├── right_*.jpg
        └── ...
```

CSV beklenen sütunlar:

```text
center,left,right,steering,throttle,reverse,speed
```

Mevcut eğitim akışı varsayılan olarak merkezi kamera görüntüsünü kullanır. Görüntü
ön işleme adımları şunlardır:

- üst kısımdan kırpma
- RGB -> YUV dönüşümü
- Gaussian blur
- 200x66 boyutuna yeniden boyutlandırma
- piksel değerlerini [0, 1] aralığına normalize etme

## Model eğitimi

Proje kök dizininden:

```bash
python training/behavioral_cloning.py --data-dir Data/Tracks --output model.keras
```

Ek seçenekler:

```bash
python training/behavioral_cloning.py \
  --data-dir Data/Tracks \
  --output model.keras \
  --epochs 10 \
  --batch-size 100 \
  --steps-per-epoch 300 \
  --validation-steps 200 \
  --no-plot
```

Bu komut:

- `driving_log.csv` okur
- direksiyon dağılımını dengelemeye çalışır
- eğitim/validasyon ayrımı yapar
- CNN modelini oluşturur
- modeli eğitir
- modeli `--output` için verilen yere kaydeder
- isteğe bağlı olarak loss ve steering dağılımı grafiklerini gösterir

## Simülatörde sürüş

Simülatör çalışırken, modelin çalıştırıldığı ana betik `drive.py` dosyasıdır.

### Standart davranışsal klonlama modu

```bash
python drive.py behavioral_cloning --model model.keras
```

Görselleştirme açıkken:

```bash
python drive.py --v behavioral_cloning --model model.keras
```

Notlar:

- simülatörün çalışıyor olması gerekir
- model yolu `--model` argümanıyla verilmelidir
- sunucu varsayılan olarak 4567 portunu dinler
- görsel işleme `img_preprocess()` fonksiyonunda yapılır

## DAgger desteği

Proje ayrıca `dagger.py` içinde DAgger-benzeri bir yardımcı içerir. Bu yardımcı,
kullanıcı manuel olarak dönüştürürken örnekleri kaydeder.

Örnek kullanım:

```bash
python drive.py dagger --model dagger_model.keras --dagger-data-dir dagger_data
```

Bu mod, modelin hata yaptığı ya da açıktan uzak kaldığı durumlarda uzman
düzeltme verisi toplamak için kullanılır.

## Model mimarisi

`training/utils.py` içinde tanımlanan model, NVIDIA benzeri bir CNN yapısına
dayanır. Temel bileşenler:

- Conv2D katmanları
- Stride içeren evrişim katmanları
- ELU aktivasyonu
- Flatten
- Dense katmanları
- Son katmanda tek çıkış (steering angle)

Kayıp fonksiyonu genelde `mse` olarak kullanılır ve optimize edici `Adam` olur.

## Görselleştirme

`visualize.py` dosyası, CNN katmanlarının aktivasyonlarını veya kernel
görünümlerini gösterir. Bu mod, `drive.py` içinde `--v` flag ile açılır.

Örnek:

```bash
python drive.py --v behavioral_cloning --model model.keras
```

## Önemli notlar

- Eğitim ve çıkarım sırasında aynı ön işleme adımları kullanılmalıdır.
- Simülatörde düşük hızda test yapılmalıdır.
- Büyük veri setleri ve model dosyaları gerektiğinde Git takip dışına alınmalıdır.
- Kod tabanında birden fazla `model` çıktısı dosyası olabilir; aktif kullanım,
  `--output` veya `--model` argümanlarına göre belirlenir.
- Bu proje şu an için "gözetimli davranışsal klonlama" yöntemini temel alır;
  tam DAgger yeniden eğitim döngüsü otomatik hale gelmemiştir.

## Hızlı başlangıç

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python training/behavioral_cloning.py --data-dir Data/Tracks --output model.keras
python drive.py behavioral_cloning --model model.keras
```

Bu akış, mevcut kod tabanında temel eğitim ve simülatör çalıştırma yoludur.
Bi de readme yazmaya üşendiğim için ai ya yazdırıyom.
