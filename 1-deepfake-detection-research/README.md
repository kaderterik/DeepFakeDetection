# 🧠 Deepfake Detection Research
### Transfer Öğrenme Kullanarak Deepfake Tespiti: FaceForensics++ Üzerinde Hiperparametre ve Model Mimarisi Analizi

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab)](https://colab.research.google.com)
[![Dataset](https://img.shields.io/badge/Dataset-FaceForensics++-green)](https://github.com/ondyari/FaceForensics)
[![Status](https://img.shields.io/badge/Status-Tamamlandı-brightgreen)]()

---

## 📌 Proje Özeti

Bu araştırma projesi, FaceForensics++ C23 veri setinde **5 kontrollü deney** ile deepfake tespitini etkileyen faktörleri sistematik olarak incelemektedir. Her deneyde tek bir değişken izole edilerek etkisi ölçülmüştür.

> **Ana Bulgu:** Öğrenme oranını 0.001'den 0.0001'e düşürmek, veri artırımı veya farklı optimizer kullanmaktan çok daha etkili olmuş; Specificity %69'dan **%88'e** yükselmiştir.

---

## 📊 Deney Sonuçları

| Deney | Değiştirilen Parametre | Accuracy | Specificity | F1 | ROC-AUC |
|-------|----------------------|----------|-------------|-----|---------|
| EXP001 | Baseline (ResNet18, Adam, LR=0.001) | 91.84% | 69.54% | 95.22% | 0.950 |
| EXP002 | + Data Augmentation | 88.99% | 36.51% ⬇️ | 93.82% | 0.911 |
| EXP003 | LR = 0.0001 ⭐ | **94.12%** | **88.51%** ⬆️ | **96.49%** | **0.978** |
| EXP004 | Adam → AdamW | 91.65% | 52.51% | 95.25% | 0.944 |
| EXP005 | ResNet18 → EfficientNet-B0 | **94.42%** | 85.90% | **96.69%** | 0.977 |

> ⭐ EXP003 en iyi maliyet-performans dengesi &nbsp;|&nbsp; EXP005 en yüksek ham doğruluk

---

## 🗂️ Klasör Yapısı

```
deepfake-detection-research/
├── 📓 02_03_Dataset_ve_EDA.ipynb   # Veri seti keşif analizi (EDA)
├── 📄 README.md                    # Bu dosya
├── 📄 .gitignore
├── ⚙️ configs/                     # Deney parametreleri (YAML)
│   ├── EXP001.yaml
│   ├── EXP002.yaml
│   ├── EXP003.yaml
│   ├── EXP004.yaml
│   └── EXP005.yaml
├── 📈 reports/                     # PDF sonuç raporları (600 DPI)
│   ├── 04_EXP001_Sonuclar.pdf
│   ├── 05_EXP002_Sonuclar.pdf
│   ├── 06_EXP003_Sonuclar.pdf
│   ├── 07_EXP004_Sonuclar.pdf
│   └── 08_EXP005_Sonuclar.pdf
└── 📊 results/
    └── experiment_log.csv          # Tüm deney metrikleri (28 sütun)
```

---

## 🧪 Deneysel Metodoloji

### Tek Değişkenli İzolasyon
Her deneyde **yalnızca bir** hiperparametre veya mimari özellik değiştirilmiş, diğer tüm ayarlar sabit tutulmuştur.

### Veri Seti
- **FaceForensics++ C23** — Önceden kırpılmış yüz görüntüleri
- Sınıf dağılımı: ~%15 Gerçek, ~%85 Sahte (**1:6 dengesizlik**)
- Görüntü boyutu: 224×224, ImageNet normalizasyonu

### Ortam
- Platform: Google Colab (NVIDIA GPU / CUDA)
- Framework: PyTorch + torchvision
- Her deney: 3 epoch, batch size: 32

---

## 🔑 Ana Bulgular

### 1. Veri Artırımı Paradoksu 🚨
Cropped yüzlere uygulanan basit augmentasyonlar Specificity'yi dramatik düşürdü:
- **%69.54 → %36.51** (EXP001 → EXP002)

### 2. Öğrenme Oranı Etkisi ✅
LR'nin 0.001'den 0.0001'e düşürülmesi sınıf dengesizliğini çözdü:
- **%69.54 → %88.51** Specificity (EXP001 → EXP003)
- **0.950 → 0.978** ROC-AUC

### 3. Maliyet-Performans Dengesi ⚖️
EfficientNet-B0 en yüksek doğruluğu verdi (%94.42) ancak eğitim süresi **%46 arttı** (35.84 dk vs ~25 dk).
**Öneri:** Gerçek zamanlı uygulamalar için **ResNet18 + LR=0.0001** kombinasyonu.

---

