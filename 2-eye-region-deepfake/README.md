# 👁️ Eye-Region Deepfake Detection

### Transfer Öğrenme Kullanılarak Göz Bölgesi Odaklı Deepfake Tespiti ve Karşılaştırmalı Model Analizi

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-orange)](https://developers.google.com/mediapipe)
[![Dataset](https://img.shields.io/badge/Dataset-FaceForensics++-green)](https://github.com/ondyari/FaceForensics)
[![Status](https://img.shields.io/badge/Status-Tamamlandı-brightgreen)]()

---

## 📌 Proje Özeti

Bu araştırma projesi, tüm yüzü analiz eden klasik deepfake tespit algoritmalarının arka plan ve ten rengine "ezberleme (overfitting)" yapmasını engellemek amacıyla **sadece göz bölgesine odaklanan** izole bir tespit sistemi sunmaktadır.

Google MediaPipe 468 noktalı FaceMesh kullanılarak FaceForensics++ C23 veri setindeki yüzlerden sadece sağ ve sol gözler kırpılarak (crop) 128x128 boyutunda bir veri seti oluşturulmuştur. Bu veri seti üzerinde tek değişkenli izolasyon prensibiyle 5 farklı modern CNN mimarisinin (ResNet18, EfficientNet-B0, DenseNet121, MobileNetV3-Small, Xception) performansı ve açıklanabilirliği (Grad-CAM) karşılaştırmalı olarak analiz edilmiştir.

> **Ana Bulgu:** Göz gibi dar uzamsal alanlarda mikro-piksellenme hatalarını (deepfake artifact) bulmak, sıradan obje tanımadan daha karmaşık bir süreçtir. "Depthwise Separable Convolutions" kullanan **Xception** mimarisi en yüksek doğruluk (%61.96) ve ROC-AUC (0.665) skorlarını elde ederek Şampiyon Model olmuştur. MobileNet gibi hafif siklet modeller ise bu ince detayları yakalamakta yetersiz kalmıştır.

---

## 📊 Deneysel Analiz ve Model Karşılaştırma Sonuçları

Tüm modeller 10 Epoch, Adam Optimizatörü (LR=0.0001) ve Batch Size=32 ayarları sabit (kontrol değişkeni) tutularak eğitilmiştir. Elde edilen test seti (2771 unseen fotoğraf) sonuçları:

| Model               | Accuracy   | Precision  | Recall | Specificity | F1-Score | ROC-AUC    | PR-AUC     | Eğitim Süresi |
| ------------------- | ---------- | ---------- | ------ | ----------- | -------- | ---------- | ---------- | ------------- |
| **ResNet18**        | %60.84     | %57.18     | %63.52 | %58.51      | %60.18   | 0.6503     | 0.6334     | ~5 Dakika     |
| **EfficientNet-B0** | %59.51     | %56.98     | %53.45 | **%64.80**  | %55.16   | 0.6361     | 0.5971     | ~6.5 Dakika   |
| **DenseNet121**     | %60.23     | %57.37     | %57.01 | %63.04      | %57.19   | 0.6455     | 0.6093     | ~10.5 Dakika  |
| **MobileNetV3-S**   | %57.42     | %54.58     | %51.20 | %62.84      | %52.84   | 0.6017     | 0.5600     | ~5.5 Dakika   |
| **Xception**        | **%61.96** | **%60.15** | %54.38 | **%68.58**  | %57.12   | **0.6650** | **0.6336** | ~13 Dakika    |

---

## 🔬 Açıklanabilir Yapay Zeka (Grad-CAM)

Modellerin "Sahte (Fake)" kararını neye göre verdiği Grad-CAM ısı haritaları ile doğrulanmıştır. Görseller, modellerin kararlarını arka plan veya rastgele piksellere göre değil, **doğrudan iris sınırlarına ve göz kapağı gölgelerindeki bozulmalara** bakarak verdiğini ispatlamaktadır.

---

## 🗂️ Klasör Yapısı

```
2-eye-region-deepfake/
├── 📄 README.md                    # Bu proje sayfası
├── 📄 .gitignore                   # Git yoksayma dosyası
├── 📄 deney2.py                    # Eğitim, test ve MediaPipe ekstraksiyon kodları
├── 📄 DeepfakeAnalizi2.pdf         # Proje Detay Raporu
├── 📂 reports/                     # Model ROC, Karmaşıklık Matrisi ve Grad-CAM çıktıları
├── 📂 data/                        # (Klasör) Orijinal ve kırpılmış veri setleri (Git ignored)
└── 📂 models/                      # (Klasör) Eğitilmiş model ağırlıkları (.pth) (Git ignored)
```

---

## 🧪 Deneysel Metodoloji

1. **Eşit Aralıklı Zaman Örneklemesi (Uniform Temporal Sampling):** Videolardan rastgele kareler almak yerine `np.linspace` ile başından sonuna kadar eşit aralıklı 20 durak noktası seçilmiştir. Böylece deepfake hatasının bulunduğu "an" kaçırılmamıştır.
2. **Video Bazlı Veri Sızıntısı (Data Leakage) Önlemi:** Aynı kişiye ait fotoğrafların hem eğitim hem test setine düşüp modelin ezber (overfitting) yapmasını engellemek için, Train/Val/Test bölüştürmesi fotoğraflar üzerinden değil **video düzeyinde** yapılmıştır. Test setindeki yüzler, modelin eğitimde hiç görmediği yepyeni yüzlerden oluşmaktadır.

---
