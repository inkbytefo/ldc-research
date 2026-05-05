# LDC v2 Yol Haritası
## Latent Diffusion Cognition — Adım Adım Geliştirme ve Doğrulama Planı

> **Sürüm:** 1.0
> **Tarih:** 2026-05-05
> **Kapsam:** Sıfırdan, küçük ölçekli, doğrulanabilir bir LDC v2 prototipi
> **Süre tahmini:** Tam zamanlı çalışmayla 14-22 hafta (3-5 ay), part-time 28-40 hafta
> **Hedef:** Kompozisyonel/akıl yürütme görevlerinde aynı parametre bütçesindeki transformer baseline'ını **istatistiksel olarak anlamlı şekilde** geçen küçük (1-10M parametre) bir model

Bu doküman aşağıdaki dokümanların sentezidir; çelişki olduğunda en yeni karar (LDC v2 — `ldc-dogrulama-raporu.md` Bölüm 4) geçerlidir:

- `latent-diffusion-cognition-mimarisi.md` — orijinal vizyon
- `ldc-dogrulama-raporu.md` — literatür + kararlaştırılmış v2 tasarımı
- `ldc-gelistirme-rehberi.md` — pratik mühendislik rehberi
- `ne-yaptik-durust-ozet.md` — neyin doğrulandığı / neyin doğrulanmadığı
- `CLAUDE-CODE-BRIEFING.md` — beklenen iş akışı

---

## 0. Yönetici Özeti

### 0.1 Hedef tek cümle

> Aynı parametre/veri bütçesinde, **graf-yapılı concept space + hibrit (Öklid + 3-5D hiperbolik) manifold + adaptive-depth latent diffusion + autoregressive decoder** kombinasyonunun, kompozisyonel genelleme ve hiyerarşik akıl yürütme görevlerinde transformer baseline'ını ölçülebilir biçimde geçip geçmediğini empirik olarak belirlemek.

### 0.2 Başarı tanımı (proje seviyesi)

| Sonuç | Anlamı | Aksiyon |
|---|---|---|
| LDC v2 baseline'ı SCAN "add jump" splitinde **≥10 puan** geçer ve ablation'lar hangi bileşenin katkıda bulunduğunu gösterir | **Yeşil:** Mimari hipotez ölçek küçük olsa da yaşıyor | Faz 7+8'e devam, workshop paper hedefle |
| LDC v2 baseline'a yakın ama belirsiz fark verir | **Sarı:** Konsept ölü değil ama kanıt yetersiz | Tek bir bileşeni (graf veya adaptive) izole et, daha fazla seed ile tekrar et |
| LDC v2 baseline'ı geçemez veya 5x'ten fazla yavaşsa | **Kırmızı:** Mevcut formülasyonla bu ölçekte çalışmıyor | Faz 5'i durdur, başarısızlık raporu yaz, alternatifleri (CALM, SONAR-LLM tarzı saf AR) tartış |

### 0.3 Yol haritasının şekli

```
Faz 0: Hazırlık ve Literatür           (1-2 hafta)  → Gate G0
Faz 1: Doğrulama Platformu             (2-3 hafta)  → Gate G1
Faz 2: Transformer Baseline            (1 hafta)    → Gate G2
Faz 3: LDC v2 Minimum Çalışan Sürüm    (2-3 hafta)  → Gate G3
Faz 4: Ablation Çalışmaları            (1-2 hafta)  → Gate G4 (KRİTİK)
Faz 5: Geometrik Zenginleştirme        (2-4 hafta)  → Gate G5
Faz 6: Adaptive Computation            (1-2 hafta)  → Gate G6
Faz 7: Genişletilmiş Benchmark'lar     (2-3 hafta)  → Gate G7
Faz 8: Raporlama ve Yayın              (1-2 hafta)  → Final
```

Her gate'te Go/Pivot/No-Go kararı verilir. Kırmızı bir gate ana hipotezi sorgulamadır.

---

## 1. Mimari Karar Özeti (Donmuş Sürüm)

`ldc-dogrulama-raporu.md` Bölüm 4'te kararlaştırılan **LDC v2** tasarımı projenin başlangıç noktasıdır. Faz 4'ten önce mimari değişikliği yasaktır.

```
[Doğal dil girdi]
        ↓
Transformer Encoder (standart)
        ↓
Concept Graph (V, E):
  V = K adet kavram slotu
  E = tipli ilişkiler (öğrenilebilir relation matrisi)
  Geometri: Öklid bileşen (R^d_e) + Hiperbolik bileşen (Poincaré ball, 3-5D)
        ↓
Latent Diffusion Core:
  - Graph transformer denoiser (manifold-aware attention)
  - Adaptive depth (halt-token / uncertainty threshold)
  - Eğitimde sabit T, çıkarımda öğrenilmiş halting
        ↓
Refined Concept Graph C*
        ↓
Autoregressive Transformer Decoder (cross-attention C* üzerinden)
        ↓
[Doğal dil çıktı]
```

### 1.1 Açık tasarım parametreleri (Faz 1'de seçilecek)

| Parametre | Faz 1 önerisi | Notlar |
|---|---|---|
| Toplam param bütçesi | 1-2M (Faz 1-2), 5-10M (Faz 5+) | Baseline ile **eşit** olmak şart |
| Slot sayısı K | 8 | SCAN için yeterli, MATH için artırılabilir |
| Öklid latent boyut d_e | 64 | |
| Hiperbolik latent boyut d_h | 0 (Faz 1'de devre dışı), 4 (Faz 5'te aktif) | |
| Diffusion adım T | 20 (sabit, Faz 1-4) | |
| İlişki tipi sayısı | 4 (Faz 1), 8 (Faz 5+) | |
| Tokenizer | byte-level BPE veya SCAN için karakter | |

### 1.2 Faz 1'de **dahil edilmeyenler** (sonra eklenecek)

- Hiperbolik bileşen (Faz 5)
- Toroidal bileşen (kapsam dışı, v3'e ertelendi)
- Topolojik perturbasyon, ilişki belirsizleştirme (Faz 5)
- Adaptive halting (Faz 6)
- Çift-katmanlı diffusion (kapsam dışı — `ldc-dogrulama-raporu.md` AR decoder kararı)

Bu sıralama bilinçli: önce çalışan iskelet, sonra zenginleştirme. `ldc-gelistirme-rehberi.md` Bölüm E Hata 6'ya bağlı.

---

## 2. Faz 0 — Hazırlık ve Literatür (1-2 hafta)

### 2.1 Çıktılar

- [ ] `docs/literature-review.md` (2-3 sayfa, `CLAUDE-CODE-BRIEFING.md` Faz 0'da listelenen 6 makaleyi kapsar)
- [ ] Ortam kurulu: PyTorch + geoopt + Hydra + wandb + pytest
- [ ] `pyproject.toml` veya `requirements.txt` kilitlendi
- [ ] Wandb projesi açık ve test run'ı yeşil
- [ ] GPU erişimi onaylı (hangi sağlayıcı, ne kadar bütçe)

### 2.2 Adımlar

1. Aşağıdaki makalelerin abstract + method + sonuç bölümlerini oku, `literature-review.md` içinde tablo halinde özetle:
   - Meta LCM (2024)
   - SONAR-LLM (arXiv 2508.05305, 2025)
   - HELM (NeurIPS 2025)
   - Coconut (arXiv 2412.06769)
   - LaDiR (arXiv 2510.04573)
   - "Diffusion Beats AR in Data-Constrained Settings" (CMU, 2025)
2. Her makaleden çıkar: (a) bizim tasarımla örtüşme, (b) farklılık, (c) bizi nereden uyarıyor.
3. `literature-review.md` sonunda **3 büyük risk** ve **2 ölçek-küçültme önerisi** yaz.
4. `requirements.txt` dosyasını oluştur, `make setup` ile tek komutta env kurulumu sağla.
5. Wandb'de `ldc-research` projesi aç. `tests/test_wandb_smoke.py` ile log testi geçir.

### 2.3 G0 Gate kriterleri

- Ortam yeniden üretilebilir (clean clone'da `make setup && make test` yeşil olmalı)
- Literatür taramasında "fatal flaw" (bizim tasarımı baştan çürüten bir sonuç) bulunmamış
- GPU bütçesi tanımlı

**Eğer G0 kırmızı:** Literatürde ölümcül bulgu varsa (örn. yeni bir paper graf-yapılı concept diffusion'ın aynı baseline'a yenildiğini göstermişse) ana tasarım sorgulanır, kapsam yeniden tartışılır.

---

## 3. Faz 1 — Doğrulama Platformu (2-3 hafta)

### 3.1 Çıktılar

`CLAUDE-CODE-BRIEFING.md` Faz 1'deki dizin yapısı kurulmuş ve şu modüller iskeleti yazılı, geçen testlere sahip:

```
ldc-research/
├── src/
│   ├── baselines/transformer.py
│   ├── ldc/
│   │   ├── encoder.py
│   │   ├── concept_space.py     # geometry registry (Öklid başlangıçta)
│   │   ├── diffusion.py         # DDPM core (sabit T, Gaussian)
│   │   ├── graph_denoiser.py    # graf transformer
│   │   ├── decoder.py           # AR transformer decoder
│   │   └── model.py             # uçtan uca LDCModel
│   ├── data/
│   │   ├── scan.py
│   │   └── synthetic.py
│   ├── train.py                 # Hydra + wandb
│   └── eval.py
├── experiments/configs/
│   ├── baseline_scan.yaml
│   └── ldc_v2_scan.yaml
├── tests/
│   ├── test_concept_space.py
│   ├── test_diffusion_step.py
│   ├── test_graph_denoiser_shape.py
│   ├── test_scan_loader.py
│   └── test_train_loop_smoke.py  # 1-batch overfit testi
└── results/
```

### 3.2 Adımlar

1. **Veri loader (SCAN):** `data/scan.py`. "simple", "add_jump", "around_right" splitlerini çekip token/karakter'e çevirir. 100 örneklik dump'ı `data/sample_scan.txt`'e yaz, gözle doğrula.
2. **Baseline skeleton:** 6 katmanlı standart encoder-decoder transformer, ~1M param. Sadece skeleton; eğitim Faz 2.
3. **LDC iskelet:**
   - `concept_space.py`: K slotlu Öklid latent, projeksiyon ve normalizasyon. (Hiperbolik kanca var ama disabled.)
   - `diffusion.py`: DDPM forward (Gaussian, sabit T=20), reverse step. Sadece **Öklid** bileşen üstünde.
   - `graph_denoiser.py`: K slotu node, öğrenilebilir relation matrisi (R x K x K), mesaj geçişli graph transformer. Cross-attention encoder context'inden.
   - `decoder.py`: Standart AR transformer, cross-attention C* üzerinde.
   - `model.py`: Yukarıdakileri birleştirir; eğitim modunda L_diff + L_lm, çıkarımda T iterasyon + AR decode.
4. **Eğitim döngüsü:** Hydra config + wandb logging. Mixed precision, grad clip, lr warmup.
5. **Birim testler:**
   - DDPM forward'ın `q(x_T) ≈ N(0, I)` olduğunu kontrol et (Faz 1 Öklid hali için)
   - 4 örneklik mini-batch'te overfit edebildiğini göster (loss → ~0)
   - Graf denoiser permütasyon testleri (slot sırası değişirse çıktı tutarlı dönüştürülüyor mu)
   - Reproducibility: aynı seed → aynı sonuç

### 3.3 G1 Gate kriterleri

- Tüm pytest yeşil
- Smoke training run'ı (1 epoch SCAN simple, küçük model) wandb'de loss düşüşü gösteriyor
- Baseline ve LDC v2 **aynı** parametre sayısına sahip (±5% tolerans, kayıt altında)
- Tek komutla yeniden çalışıyor: `python -m src.train experiment=ldc_v2_scan`

**Eğer G1 kırmızı:** Mühendislik problemi, mimari kararla ilgisiz. Çözmeden Faz 2'ye geçme.

---

## 4. Faz 2 — Transformer Baseline (1 hafta)

`CLAUDE-CODE-BRIEFING.md`: "Do not skip the baseline." Baseline iyi tune edilmezse her LDC iddiası zayıf baseline'a karşıdır.

### 4.1 Çıktılar

- [ ] Baseline tam SCAN train + 3 değerlendirme split'inde (`simple`, `add_jump`, `around_right`)
- [ ] Her split için ≥3 seed, ortalama ± std raporu
- [ ] `results/baseline_scan.md` numerik tablolar + wandb run linkleri
- [ ] Hiperparametre tarama loğu (`experiments/sweeps/baseline_hpo.yaml`)

### 4.2 Adımlar

1. Baseline için en az 12 noktalı bir hiperparametre taraması yap: lr ∈ {1e-4, 3e-4, 1e-3}, dropout ∈ {0.0, 0.1, 0.3}, warmup ∈ {1k, 5k}. Tek seed ile en iyiyi bul.
2. En iyi hiperparametre ile **3 seed** çalıştır.
3. Her split için: token accuracy, sequence accuracy, eğitim eğrisi grafiği.
4. Beklenen aralık: `add_jump` ~ %0-30 (literatür); bu aralığa düşmüyorsak baseline bozuk demektir.

### 4.3 G2 Gate kriterleri

- `add_jump` baseline accuracy literatür aralığında (~%0-30, paramı küçük tutuyoruz)
- Standard split %95+ (yoksa overfit/underfit/data sorunu var)
- Eğitim sürecinde NaN/instability yok

**Eğer G2 kırmızı:** Baseline'ı düzeltmeden LDC'ye geçme. LDC'nin "üstün gelmesi" anlamsız olur.

---

## 5. Faz 3 — LDC v2 Minimum Çalışan Sürüm (2-3 hafta)

### 5.1 Çıktılar

- [ ] LDC v2 (Öklid-only, sabit T=20, graf-yapılı) SCAN'in tüm 3 split'inde eğitilmiş
- [ ] ≥3 seed
- [ ] `results/ldc_v2_phase3.md`: baseline'a karşı kafa-kafaya karşılaştırma
- [ ] Eğitim ek maliyeti: per-step süre, toplam eğitim süresi, parametre sayısı ölçülmüş

### 5.2 Eğitim sırası (`ldc-gelistirme-rehberi.md` B5'ten uyarlanmış)

1. **Aşama A (1-2 epoch):** Encoder+Decoder'ı diffusion olmadan, K=1 slot ile autoencoder gibi eğit. Latent'in mantıklı organize olduğunu doğrula (cosine sim heatmap).
2. **Aşama B (ana eğitim):** Encoder/Decoder'ı bırakıp tüm sistemi joint eğit. Loss = L_diff (DDPM standart) + α · L_lm (decoder cross-entropy). α = 1.0 başlangıç, ablation'da değişir.
3. **Aşama C (kısa fine-tune):** Düşük lr ile end-to-end son ince ayar.

### 5.3 G3 Gate kriterleri

- LDC v2 eğitimi stabil (NaN yok, loss eğrisi inişli)
- LDC v2 standart split'te baseline'a yakın (≥%90 baseline'ın değerinin)
- Per-step süre baseline'ın **5x'inden** fazla değil (aksi halde Faz 5+'te yaşanabilir değil)

**Eğer G3 kırmızı:**
- Stabil değilse: graf denoiser'a layer-norm, lr ölçeğini düşür, T'yi 10'a indir
- Standart split'te kötüyse: tasarım hatası muhtemel; encoder/decoder'ın diffusion'a sahip olmadığı bir oblation çalıştır, asıl sorun nerede tespit et
- 5x'ten yavaşsa: çıkarımda T'yi düşür, kısa-bypass yolu ekle

---

## 6. Faz 4 — Ablation Çalışmaları (1-2 hafta) — KRİTİK FAZ

`CLAUDE-CODE-BRIEFING.md`: "Do not present optimistic results without ablations." Bu faz hangi bileşenin gerçekte çalıştığını söyler.

### 6.1 Çıktılar

`results/ablations.md`: aşağıdaki tüm varyantların matrisi, her hücre 3 seed.

| Varyant | Slot sayısı | Diffusion var mı? | Açıklama |
|---|---|---|---|
| A0: Baseline | — | — | Faz 2 |
| A1: LDC v2 (full) | 8 | T=20 | Faz 3 |
| A2: K=1 (LCM-benzeri) | 1 | T=20 | Graf yapısının katkısı |
| A3: T=1 | 8 | T=1 | Diffusion'ın katkısı (encoder/decoder + tek graf forward) |
| A4: T=20 sabit, encoder context devre dışı | 8 | T=20 | Diffusion'ın conditioning'siz çalışıp çalışmadığı |
| A5: A1 + 2x param | 8 | T=20 | Avantaj parametre mi yoksa mimari mi? |

### 6.2 Yorumlama matrisi

| Bulgu | Yorum |
|---|---|
| A1 > A2 | Graf yapısı kazandırıyor (yeni katkımız) |
| A1 > A3 | Diffusion gerçekten yardım ediyor |
| A1 ≈ A3 | Diffusion gereksiz; mimari basitleştirilmeli |
| A1 ≈ A2 | Graf yapısı katkı vermiyor; tasarımın özgün iddiası zayıf |
| A1 ≈ A5 (baseline'ı geçen sürüm) | Sadece parametre değil, mimari avantaj var |

### 6.3 G4 Gate (KRİTİK)

- A1, A0'ı `add_jump` üzerinde **istatistiksel olarak anlamlı** (Welch t-test p<0.05 veya 3 seed'de aralık çakışmıyor) geçiyor mu?
- A1, A2 ve A3'ü geçiyor mu?
- A1 ≈ A5 ise: avantaj mimari (iyi)

**Eğer G4 kırmızı (yani A1 baseline'ı geçemezse):**
- **Bu projenin en önemli kararı buradadır.** `ne-yaptik-durust-ozet.md` ruhuyla davran.
- 1 hafta hızlı bir "post-mortem": implementation hatası, hiperparametre, scale'ın düşük olması.
- Hala kırmızıysa: Faz 5'i durdur. `results/honest_report.md` yaz: "Bu ölçek ve formülasyonda LDC v2 baseline'ı geçmiyor." Sonra istersen pivot et (Faz 5 yerine farklı bir niş, örn. data-constrained regime, veya CALM benzeri yaklaşım).

---

## 7. Faz 5 — Geometrik Zenginleştirme (2-4 hafta)

**Sadece G4 yeşilse.** Hiperbolik bileşeni ekle.

### 7.1 Çıktılar

- [ ] `concept_space.py` Poincaré ball desteği (3-5D), `geoopt` üzerinden
- [ ] Riemannian Adam optimizer entegrasyonu
- [ ] Manifold-aware attention `graph_denoiser.py` içinde
- [ ] WordNet alt kümesi veya yeni hiyerarşik probe testi (concept space'in gerçekten ağaç gibi organize olduğunu gösteriyor mu)
- [ ] `results/hyperbolic_ablation.md`: H1 (Öklid-only), H2 (Öklid + 3D hiperbolik), H3 (Öklid + 5D hiperbolik)

### 7.2 Risk azaltma

`ldc-dogrulama-raporu.md` Bölüm 1.4'teki uyarıyı dikkate al: **numerik hassasiyet**.

- fp32 zorunlu (mixed precision'ı hyperbolic operasyonlarda kapat)
- `geoopt` projeksiyon clipping'i daima açık
- Norm > 1-ε durumlarını `tests/test_poincare_stability.py` ile yakala
- 2D'de çalışmıyor — 3D minimum (Test 1 sonucu)

### 7.3 G5 Gate

- Hiperbolik versiyonu Öklid-only versiyondan **istatistiksel olarak iyi** mi?
- Iyileşme yoksa: hesap maliyetine değmez, hiperbolik bileşeni v3'e ertele, Faz 6'ya geç.

---

## 8. Faz 6 — Adaptive Computation (1-2 hafta)

### 8.1 Çıktılar

- [ ] Halt-network eklendi (her diffusion adımı sonrası confidence skoru)
- [ ] L_total = L_task + λ · n_steps; λ tarama ile seçilmiş
- [ ] `results/adaptive.md`: kolay vs zor örneklerde gerçekten farklı n_steps oluşuyor mu (histogram)
- [ ] Çıkarım hızlanması (örnek başına ortalama T) ölçülmüş

### 8.2 G6 Gate

- Doğruluk düşmeden ortalama T düşüyor mu (örn. T_eff < 15)?
- Zor örnekler (uzun cümleler, kompozisyonel split) **otomatik** olarak daha çok adım kullanıyor mu?

Bu, `ne-yaptik-durust-ozet.md`'de en güçlü matematiksel motivasyona sahip iddia. Faz 6 mimarinin özgün bilimsel hikayesinin omurgası.

---

## 9. Faz 7 — Genişletilmiş Benchmark'lar (2-3 hafta)

`ldc-dogrulama-raporu.md` Bölüm 4.2'deki niş hedefler.

### 9.1 Önerilen sıra

1. **COGS** veya **gSCAN** — kompozisyonel zorluk artırma (SCAN extension)
2. **GSM8K (küçük subset)** — hiyerarşik akıl yürütme. Model 5-10M ile bu görevde tam çözüm yapamaz; metrik: solution_format_acc + step_acc
3. **Veri-kısıtlı transformer baseline karşılaştırması** — eğitim datasının %10/%30/%100'ünde her iki model. CMU bulgusu: az veride diffusion kazanır. Bu hipotezi test ediyoruz.

### 9.2 G7 Gate

- En az **bir** benchmark'ta LDC v2 baseline'ı net geçti
- Hesap maliyeti detaylıca raporlanmış (FLOPs, wall-clock, peak memory)
- Hangi görevde geride kaldığı ve **nedeni** açık (`results/failure_analysis.md`)

---

## 10. Faz 8 — Raporlama ve Yayın (1-2 hafta)

### 10.1 Çıktılar

- [ ] `results/report.md` — `CLAUDE-CODE-BRIEFING.md` Faz 3'teki tüm kalemler:
  - Tüm sayısal sonuçlar tablo halinde
  - Loss eğrileri ve eğitim dinamikleri
  - Dürüst değerlendirme (kazandı mı, kaybetti mi, ne kadar)
  - Failure analizi
  - Maliyet analizi (per-step, per-sample)
  - Devam/pivot/dur tavsiyesi
- [ ] arXiv-ready preprint draft (8 sayfa, NeurIPS template)
- [ ] Kod dokümantasyonu, README güncellenmiş, MIT/Apache lisans yerleştirilmiş
- [ ] Tüm wandb run'ları halka açık veya en az bağlanmış

### 10.2 Yayın hedefleri (gerçekçi)

`ldc-dogrulama-raporu.md` Bölüm 5.3: "ICLR/NeurIPS workshop seviyesi" gerçekçi hedef. Faz 7 sonuçları çok güçlüyse main track. Sonuç negatifse "lessons learned" workshop'u veya teknik blog.

---

## 11. Risk Kaydı

| ID | Risk | Olasılık | Etki | Hafifletme |
|---|---|---|---|---|
| R1 | LDC baseline'ı geçemez (G4 kırmızı) | **Yüksek** (`ne-yaptik-durust-ozet.md`'ye göre `kanıtlanmadı`) | Proje pivot | Faz 4 öncesi 1 haftalık post-mortem hazırlığı, dürüst rapor şablonu hazır |
| R2 | Hesap maliyeti baseline'ın 10x'i | Orta-Yüksek | Pratik kullanım yok | T'yi düşür, sparse attention, FLOPs ölçümünü Faz 3'ten itibaren her run'da yap |
| R3 | Hiperbolik numerik instability | Orta | Faz 5 başarısız | fp32, clipping, dedicated test, Öklid-only fallback hazır |
| R4 | Eğitim dengesizliği (multi-loss) | Orta | Sonuçlar gürültülü | Loss ağırlıklarını sweep'le, gradient norm'larını wandb'de izle |
| R5 | SCAN'da overfit, gerçek genelleme yok | Düşük-Orta | Yanlış pozitif | Faz 7'de COGS ve gSCAN ile teyit |
| R6 | İmplementasyon hatası: hiperbolik / graph attention'ın yanlışlığı | Yüksek (kolay subtle hata) | Yanlış sonuç | Birim testler, permütasyon invariance testleri, ikinci göz code review |
| R7 | "Yeni paper çıktı, fikrimiz scoop'landı" | Orta | Bilimsel katkı azalır | Faz 0'da literatür tarandı; her ay arXiv kontrolü |
| R8 | GPU bütçesi tükenir | Orta | Faz 7+ kesilir | Faz 0'da bütçe net, deneyleri bütçe sırasına göre prioritize et |

---

## 12. Karar Noktaları (Gate Özeti)

| Gate | Sonuç → Aksiyon |
|---|---|
| G0 | Yeşil: Faz 1. Sarı: ortam sorunu, çöz. Kırmızı: literatürde fatal flaw, kapsam revize. |
| G1 | Yeşil: Faz 2. Kırmızı: mühendislik, çöz. |
| G2 | Yeşil: Faz 3. Kırmızı: baseline'ı düzelt; LDC'ye geçme. |
| G3 | Yeşil: Faz 4. Kırmızı: stabilize et veya kapsam küçült. |
| **G4** | **Yeşil: Faz 5. Kırmızı: post-mortem + dürüst rapor + pivot/durdur kararı.** |
| G5 | Yeşil: Faz 6. Kırmızı: hiperbolik'i ertele, Faz 6'ya geç. |
| G6 | Yeşil: Faz 7. Kırmızı: adaptive'i bırak, Faz 7'ye geç. |
| G7 | Yeşil: Faz 8 (yayın). Kırmızı: dürüst negatif rapor + lessons learned. |

---

## 13. Kaynak Tahmini

| Kaynak | Min | Hedef |
|---|---|---|
| GPU | 1x A100 (Colab Pro+ veya RunPod) | 1x A100 sürekli |
| Aylık bütçe | ~$50 (Colab) | ~$300 (RunPod, 200-400 saat) |
| Disk | 50 GB (datasets + checkpoints) | 200 GB |
| Wandb | Ücretsiz tier | Ücretsiz tier (akademik) |
| İnsan zamanı (FT) | 14-22 hafta | 18 hafta nominal |
| İnsan zamanı (PT) | 28-40 hafta | — |

---

## 14. Bu Hafta İçin Somut Eylem Listesi

`ldc-gelistirme-rehberi.md` Bölüm F'ten uyarlanmış, bu yol haritasıyla uyumlu hâl:

**Pzt-Çar (Faz 0 başlat):**
1. `requirements.txt` yaz, conda env kur, `make setup` test et
2. Wandb projesi aç, smoke run logla
3. Listedeki 6 makaleden ilk 3'ünü oku (LCM, SONAR-LLM, HELM)

**Per-Cum:**
4. Kalan 3 makaleyi oku (Coconut, LaDiR, CMU diffusion)
5. `docs/literature-review.md` taslağını yaz
6. SCAN datasetini indir, 100 örnek elle oku

**Hafta sonu:**
7. DDPM paper'ını yeniden oku, anlamadığın yerleri `docs/notes/ddpm-questions.md` dosyasına yaz
8. G0 kontrol listesini geç → Faz 1 planlama oturumu

---

## 15. Uyumluluk: Bu Yol Haritası ile Mevcut Dokümanlar

| Mevcut doküman | Bu yol haritasıyla ilişki |
|---|---|
| `latent-diffusion-cognition-mimarisi.md` | Vizyon. Toroidal manifold ve çift-katmanlı diffusion **bu sürümde kapsam dışı**, v3'e ertelendi. |
| `ldc-dogrulama-raporu.md` (LDC v2) | **Donmuş tasarım**. Bu yol haritasının mimari girdisidir. |
| `ldc-gelistirme-rehberi.md` | Pratik mühendislik notları. Faz 1, 5, 6 adımları buradan uyarlandı. |
| `ne-yaptik-durust-ozet.md` | Mevcut doğrulamanın sınırları. G4'te bu ruh esastır. |
| `CLAUDE-CODE-BRIEFING.md` | Çalışma protokolü. Faz isimlendirmesi (0-3) korundu, alt-fazlara genişletildi. |

---

## 16. Son Söz

Bu yol haritası bir hipotez doğrulama planıdır, bir ürün roadmap'i değil. Hipotez **yanlış da olabilir** ve bu bilgi de değerlidir. G4'te kırmızı çıkması, projenin başarısızlığı değil — yanlış bir araştırma yönüne 12 ay yatırım yapmaktan kurtulmaktır.

Tek hedef: **doğru cevap**, övgü değil.
