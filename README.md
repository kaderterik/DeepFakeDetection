# 🧠 DeepFake Detection — BAUN CENG Research Repository

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab)](https://colab.research.google.com)
[![University](https://img.shields.io/badge/BAUN-Bilgisayar%20Mühendisliği-navy)](https://www.balikesir.edu.tr)

---

## 🎯 Bu Repo Ne İçin?

Bu depo, **Balıkesir Üniversitesi Bilgisayar Mühendisliği** bölümünde yürütülen yapay zeka ve derin öğrenme araştırmalarını barındırmaktadır.

Temel odak noktası: **Deepfake (sentetik yüz) tespiti** alanında transfer öğrenme yöntemlerinin sistematik olarak incelenmesi.

---

## 🔬 Ne Araştırıyoruz?

Sosyal medyada hızla yayılan yapay zeka üretimi sahte videolar (deepfake), kimlik dolandırıcılığından dezenformasyona kadar ciddi tehditler oluşturmaktadır. Bu repo, bu tehdide karşı geliştirilen **otomatik tespit sistemlerinin** nasıl daha doğru ve güvenilir hale getirilebileceğini araştırmaktadır.


### Temel Sorular:
- Hangi model mimarisi deepfake tespitinde daha başarılı?
- Veri artırımı (augmentation) gerçekten faydalı mı?
- Öğrenme oranı ve optimizer seçimi performansı nasıl etkiliyor?
- Sınıf dengesizliği sorunu nasıl aşılabilir?

---

## 📁 Projeler

### 🔷 [1-deepfake-detection-research](./1-deepfake-detection-research/)
FaceForensics++ C23 veri seti üzerinde **5 kontrollü deney** içeren kapsamlı araştırma projesi.

| Deney | Konu | Sonuç |
|-------|------|-------|
| EXP001 | Baseline (ResNet18) | Acc: %91.84 |
| EXP002 | Veri Artırımı | Acc: %88.99 ⬇️ |
| EXP003 | Düşük Öğrenme Oranı | Acc: %94.12 ⭐ |
| EXP004 | AdamW Optimizer | Acc: %91.65 |
| EXP005 | EfficientNet-B0 | Acc: %94.42 |

> **En kritik bulgu:** Öğrenme oranını düşürmek (LR: 0.001 → 0.0001), modelin gerçek yüzleri tanıma başarısını **%69'dan %88'e** çıkardı.

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Amaç |
|-----------|------|
| **PyTorch** | Derin öğrenme modelleri |
| **torchvision** | ResNet18, EfficientNet-B0 transfer learning |
| **Google Colab** | GPU destekli eğitim ortamı |
| **scikit-learn** | Metrik hesaplama (F1, AUC, CM) |
| **matplotlib** | Grafik ve görselleştirme |
| **FaceForensics++** | Benchmark veri seti |

---

## 👤 Araştırmacı

**Kader Terik**  
Balıkesir Üniversitesi — Bilgisayar Mühendisliği Bölümü (BAUN CENG)  
📅 2026

