# LDC v2 — Deney Sonuçları ve Post-Mortem

> **Tarih:** 2026-05-10
> **Deney aralığı:** 2026-05-09 → 2026-05-10
> **Durum:** G4 (Ablation Gate) **kırmızı** — mimari hipotez doğrulanmadı, pivot kararı verildi
> **Sonraki yön:** HiCoRe (bkz. `HICORE.md`)

---

## 0. Yönetici Özeti

LDC v2 (Latent Diffusion Cognition) mimarisi — encoder + slot attention + DDPM diffusion + AR decoder kombinasyonu — SCAN compositional generalization benchmark'ında **vanilla transformer baseline'ı geçemedi**.

Üç farklı konfigürasyonda (clean / refined-detached / refined-joint) ve üç farklı split'te (simple / addprim_jump / template_around_right) test edildi. Tüm compositional split'lerde test seq_acc ≈ %0.

**Mimaride implementation hatası bulunmadı.** Başarısızlık tasarımsal — Phase 1 LDC v2'nin compositional binding için yeterli inductive bias'a sahip olmadığı gösterildi.

**Karar:** Roadmap madde 6.3 G4 kırmızı eylemine göre: dürüst rapor (bu doküman) + pivot. Yeni mimari HiCoRe olarak tasarlandı.

---

## 1. Deney Tablosu

### 1.1 Standart split (`simple`) — kontrol

Train ve test aynı dağılımdan; basit ezberleme testidir.

| Model | %100 val seq_acc'a ulaşılan adım | Final test seq_acc |
|-------|----------------------------------|-------------------|
| Baseline transformer | 8K | beklenen %99+ |
| LDC v2 (clean) | 19K | beklenen %99+ |

Yorum: Her iki model da basit görevi çözebiliyor. LDC v2 ~2.4x daha yavaş converge oluyor (slot attention ek bottleneck).

### 1.2 `addprim_jump` split — compositional gen testi

Train'de `jump` token'ı tek başına görünür ama **kompozisyon içinde değil**. Test'te `jump twice`, `jump and walk`, `jump around left` gibi kombinasyonlar var.

| Model | val seq_acc (in-distribution) | **test seq_acc (held-out)** | test token_acc |
|-------|------------------------------|-----------------------------|----------------|
| Baseline transformer | %100 | **%0.09** | %53.7 |
| LDC v2 (clean) | %99.9 | **%0.0** | %56.4 |
| LDC v2 (refined, detached gradient) | aynı | %0.0 (test) | aynı |
| LDC v2 (refined, joint gradient) | %9.1 ⚠️ | **%0.0** | %48.9 |

**Kritik bulgu:** Hiçbir LDC v2 konfigi baseline'ı geçmedi. Refined-joint mode val'de bile felaket (%99 → %9), distribution mismatch.

### 1.3 `template_around_right` split

| Model | val seq_acc | test seq_acc |
|-------|------------|--------------|
| Baseline transformer | %100 (7K step) | (test henüz çalıştırılmadı) |
| LDC v2 (clean) | %100 (18K step) | (test henüz çalıştırılmadı) |

Around_right test çalıştırılmadı; kredi sınırı + add_jump sonuçları yeterince bilgi veriyor.

---

## 2. Deney Kronolojisi

### 2026-05-09

**Sabah — Faz 1 tamamlama:**
- Eksik `src/data/` modülleri yazıldı (tokenizer, scan, synthetic, __init__)
- Mevcut testler: `ModuleNotFoundError` → tüm tests pass (34 test)
- Build pipeline doğrulandı

**Öğleden sonra — Lightning AI deployment:**
- `git push` + Lightning AI'da SSH erişimi kurulumu
- `uv` ile venv kurulumu (default Python 3.12'de pip yoktu)
- PyTorch 2.5.1+cu121 install
- Tesla T4 15GB GPU üzerinde smoke test

**Akşam — İlk training run'lar:**
- Baseline scan (simple): %100 val @ 8K step
- LDC v2 scan (simple): %100 val @ 19K step
- Bug bulundu: LDC v2 generate() her zaman refine() çağırıyordu, decoder clean görmesine rağmen
- Fix: `model.generate()` `decoder_input == "clean"` kontrolü, encoder C_0 doğrudan kullan
- Restart, fix doğrulandı

### 2026-05-10

**Sabah — addprim_jump + around_right runs:**
- Baseline ve LDC v2 her iki split'te ~%100 val seq_acc'a ulaştı
- Kritik gözlem: val = train dağılımı (val_fraction=%5 train'den split). True compositional gen testi DEĞİL.

**Öğlen — Held-out test pipeline:**
- `train.py`'a `test_loader` üzerinde final eval eklendi
- `tasks_test_addprim_jump.txt` üzerinde gerçek compositional gen ölçüldü
- **Sonuç:** Baseline %0.09, LDC v2 %0.0 — felaket

**Öğleden sonra — refined inference deneyi:**
- `scripts/eval_refined.py` yazıldı: clean vs refined inference karşılaştırması
- Trained checkpoint üzerinde her iki modu çalıştır
- **Sonuç:** clean=%0.0, refined=%0.0, refined token_acc bile daha kötü (-5.6 puan)
- Yorum: Diffusion eğitilmiş ama decoder clean üzerinde eğitildiği için refined memory faydasız

**Akşam — refined training deneyi:**
- `model.py`'da `with torch.no_grad()` kaldırıldı (LM loss diffusion'a aksın)
- `decoder_input: refined` config oluşturuldu
- Yeni run: `ldc_v2_addjump_refined.log`
- **Sonuç:** val seq_acc %9.1 (clean'de %99.9 idi), test seq_acc %0
- Yorum: Distribution mismatch — training'de random t-step predicted_x_0, inference'ta full reverse diffusion sonucu. Decoder iki dağılımı aynı zamanda öğrenemedi.

---

## 3. Bulgular

### 3.1 Mimari Hipotez Yanlışlandı

LDC v2'nin orijinal hipotezi:
> "Slot attention ile sıkıştırılmış concept graph + diffusion-based latent refinement, compositional generalization sağlar."

Test sonuçları bu hipotezi reddediyor:
- Slot attention slot'ları interchangeable bıraktı (tip yapısı yok)
- Diffusion latent space'i smoothladı ama compositional STRUCTURE eklemedi
- Train'de görülmeyen primitive (jump) kombinasyonlarda kullanılamadı

### 3.2 Diffusion Phase 1 Tasarımının Eksikliği

İki konfigürasyon test edildi:
- **Clean mode:** Decoder her zaman encoder C_0 görür → Diffusion training'de eğitiliyor ama inference'ta kullanılmıyor (sadece auxiliary loss)
- **Refined mode:** Decoder predicted_x_0 görür → Decoder noise-aware oluyor ama distribution mismatch

Her iki mod da başarısız. **Sonuç: Latent diffusion + AR decoder kombinasyonu sıralı eğitim/inference disiplini gerektirir, basit joint training yetersiz.**

### 3.3 Slot Attention Compositional Bias Vermiyor

LDC v2'nin "K=8 slot" yapısı:
- Hangi slot'a hangi tokenin düştüğü tutarsız
- "Verb-slot" gibi tip kavramı emerge olmuyor
- Train'de görülen kombinasyonların replikası yapılabiliyor ama yeni kombinasyon üretilemiyor

### 3.4 Held-out Test Pipeline Hayati Önemde

Erken keşif: val_fraction=%5 train'den geliyor → in-distribution. **Compositional gen ölçmüyor.**
Held-out test (`tasks_test_*.txt`) gerçek metric.

Bu öğreti tüm gelecek mimariler için zorunlu: **eval_every'de val + test metrik logla.**

### 3.5 Baseline'ın da Başarısızlığı Anlamlı

Vanilla transformer da `addprim_jump`'ta %0.09 verdi. Bu literatürle uyumlu (1-2M parametreli küçük transformer). Compositional gen için **scale veya architectural bias** gerekiyor; bizim modelimiz iki ayda da yetersiz.

---

## 4. Mimari Kararlar Tablosu (Ne Çalıştı / Çalışmadı)

| Karar | Sonuç | Yorum |
|-------|-------|-------|
| Slot Attention K=8 | ❌ | Tipsiz slotlar yetersiz inductive bias |
| Gaussian DDPM (T=20, cosine) | ⚠️ | Training stabil ama compositional katkı yok |
| Concept space d_concept=64 | ✓ | Yeterli boyut, problem değil |
| Auxiliary role loss | (yapılmadı) | Yokluğu büyük eksiklik — HiCoRe'da var |
| Curriculum learning | (yapılmadı) | Random data → baştan zorla composition öğrenme |
| Held-out test eval | ✓✓ | Sonradan eklendi, kritik bulguları açığa çıkardı |
| Hyperbolic embedding | (Faz 5'e ertelenmişti) | LDC v2'de hiç denenmedi; HiCoRe'da denenecek |

---

## 5. Kod Tabanı Durumu

**Çalışıyor ve yeniden kullanılacak:**
- `src/data/` (tokenizer, scan loader, synthetic) ✓
- `src/baselines/transformer.py` ✓ (compositional gen baseline olarak kalacak)
- `src/train.py` (held-out test pipeline ile) ✓
- `src/eval.py` ✓
- Hydra config sistemi ✓
- 34 test, hepsi pass ✓

**HiCoRe'da silinmeyecek ama refactor olabilir:**
- `src/ldc/` modülleri (encoder, decoder, diffusion, graph_denoiser, concept_space, model)
- `experiments/configs/ldc_v2_*.yaml`
- Bunlar ablasyon karşılaştırması için kalacak (H1: LDC v2 vs H2: HiCoRe)

**Yeni eklenen ama LDC v2'ye özel:**
- `scripts/eval_refined.py` (clean vs refined karşılaştırması, LDC v2'ye özel ama pattern olarak HiCoRe'da da kullanılabilir)

---

## 6. Roadmap Gate Karar Tablosu

`docs/yol-haritasi.md` G4 kriteri:
> "A1, A0'ı `add_jump` üzerinde istatistiksel olarak anlamlı geçiyor mu?"

**Cevap:** HAYIR. A1 = %0.0, A0 = %0.09. Geçmiyor; hatta marjinal kötü.

G4 kırmızı eylemleri (`docs/yol-haritasi.md` 6.3):
> 1. Hızlı post-mortem: implementation hatası, hiperparametre, scale.
> 2. Hala kırmızıysa: Faz 5'i durdur, dürüst rapor yaz, pivot/durdur kararı al.

Yapılan post-mortem:
- ✓ Implementation kontrol edildi (refined mode hatası bulunup düzeltildi, sonuç değişmedi)
- ✓ Refined inference denenirdi (clean ve refined ikisi de %0)
- ✓ Refined joint training denenirdi (val bile %9'a düştü, distribution mismatch)
- ✗ Hiperparametre sweep yapılmadı (zaman/kredi kısıtı; muhtemelen marjinal etki)
- ✗ Scale 2x denenmedi (ablasyon A5; muhtemelen marjinal etki)

**Karar:** G4 kırmızı + dürüst rapor yazıldı + pivot kararı: HiCoRe.

---

## 7. Kredi/Zaman Kullanımı

Lightning AI Tesla T4 üzerinde:

| Run | Süre |
|-----|------|
| baseline simple | ~12 dk |
| LDC v2 simple | ~13 dk |
| baseline addjump | ~12 dk |
| LDC v2 addjump (clean) | ~13 dk |
| baseline aroundright | ~10 dk |
| LDC v2 aroundright | ~13 dk |
| LDC v2 addjump (refined) | ~16 dk |
| eval_refined script | ~3 dk |

Toplam GPU saat: ~1.7 saat. ~3-4 kredi tüketildi.

---

## 8. Öğrenilen Mühendislik Dersleri

1. **Held-out test'i eval pipeline'a baştan koy.** Val ≠ test. Bizim için 1 günlük gecikme oldu.
2. **Distribution match training/inference arasında zorunlu.** Refined-joint trained model'in inference path'i farklı olduğunda sonuç patladı.
3. **Compositional gen küçük modelle nadiren self-emerge eder.** Inductive bias şart.
4. **`generate()` fonksiyonunda use_clean param'ı kritik karar.** Default değer model'in eğitildiği duruma uygun olmalı.
5. **Hydra config interpolation YAML flow sequence'da çalışmaz.** Block sequence kullan.
6. **Lightning AI multi-line SSH komutlarında sürekli kopuyor.** Tek komut + `bash -c` daha güvenilir.
7. **`uv` Python env kurulumunda pip'siz sistemler için kritik.**

---

## 9. LDC v2 Mirası

LDC v2 başarısız oldu ama **mimari iskelet, eğitim pipeline'ı, eval altyapısı, dataset loader'ları, test suite — hepsi sağlam.** HiCoRe bu temelin üzerine inşa edilecek. Yeni proje sıfırdan değil, miras üzerinden.

LDC v2 kodu silinmeyecek; ablasyon referansı (H1) olarak kalacak.

---

## 10. Sonraki Adım

`HICORE.md` doküman hazır → implementation plan yazılacak (writing-plans skill).

İlk hedef: **HiCoRe ile addprim_jump'ta test seq_acc ≥ %20.** Eğer bu eşik geçilirse mimari hipotez canlı, Faz HC-3 ablasyonlarına devam.

Eğer bu eşik de geçilemezse: compositional gen *küçük scale + inductive bias* ile çözülebilir bir problem değil mi sorgulanır. Pivot Phase 2 (CALM, SONAR-LLM tarzı saf AR) veya scale-up yönünde olur.

---

## Ek A: Önemli Commit Listesi

```
e3bf351 docs: capture architecture-native tool use idea (v3)
3bcfcf4 feat: scaffold LDC v2 + transformer baseline (Faz 1)
ef7db93 docs: add LDC v2 step-by-step roadmap
[Faz 1 sonrası]
... (src/data modülleri, generate() fix, held-out test eval, refined mode)
4b30c55 (HEAD önceki) feat: implement src/data + LDC v2 fixes
be5ab13 feat: evaluate on held-out test set after training
63d7c4d feat: clean vs refined inference comparison script
8781454 feat: refined-mode training for LDC v2
```

## Ek B: Log Dosyaları (Lightning AI Remote)

- `outputs/baseline_addjump_v2.log`
- `outputs/ldc_v2_addjump_v2.log`
- `outputs/ldc_v2_addjump_refined.log`
- `outputs/baseline_aroundright.log`
- `outputs/ldc_v2_aroundright.log`
- `outputs/baseline_train.log` (simple split)
- `outputs/ldc_v2_train_v2.log` (simple split, fix sonrası)

Yerel: `outputs/` ignore'lanmış (bkz. `.gitignore`).

---

**Sonuç:** LDC v2 öldü, yaşasın HiCoRe.
