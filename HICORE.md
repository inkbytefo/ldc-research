# HiCoRe — Hierarchical Composition via Role-bound Refinement

> **Sürüm:** 0.1 (Tasarım taslağı)
> **Tarih:** 2026-05-10
> **Durum:** Brainstorm onaylandı, implementasyon planı henüz yazılmadı
> **Önceki proje:** LDC v2 (bkz. `results/2026-05-10-ldc-v2-postmortem.md`)

---

## 0. Özet

HiCoRe, **kompozisyonel ve sistematik genelleme** problemine odaklanan yeni bir mimari önerisidir. Klasik Tensor Product Representations (Smolensky 1990, Schlag 2019) zeminini modern transformer + iteratif rafineleme + hiperbolik geometri ile birleştirir.

Anahtar fikir: **dilsel kompozisyonun temeli, "role-filler binding" denilen değişken bağlamadır.** "Walk twice" ile "Jump twice" aynı yapısal şablonun iki örneğidir; sadece dolgu (filler) değişir, role aynıdır. Klasik bir LLM bu yapıyı keşfetmek zorundadır; HiCoRe bu yapıyı mimarinin içine yerleştirir.

LDC v2'nin başarısızlığı bu eksikliğin somut kanıtıdır: tipsiz slot'lar, compositional binding olmadan, train'de görülmemiş `jump twice` kombinasyonunu üretemedi (test seq_acc = %0).

---

## 1. Motivasyon — LDC v2 Neden Yetmedi

LDC v2 testlerinde üç kök neden gözlemlendi:

1. **Slot Attention'ın tip yapısı yok.** 8 slot interchangeable. Model "bu verb-slot'u" diye konuşmuyor; slot'a hangi rolün düştüğü tutarsız.
2. **Diffusion yanlış değişkeni rafinise ediyordu.** Concept position'ları değil, *yapısal binding'i* rafinise etmek lazım.
3. **Açık kompozisyonel sinyal yok.** "Twice operatörü tüm verb'lere aynı şekilde uygulanır" kuralını model kendi keşfetmek zorundaydı; küçük scale'de keşfedilmiyor.

Sonuç: LDC v2 baseline transformer'a karşı her iki konfigde (clean ve refined) compositional gen testinde **net üstünlük gösteremedi.**

HiCoRe bu üç problemin de doğrudan çözümünü mimaride taşır:

- Tipli rol envanteri (problem 1)
- Bound tensor üzerinde rafineleme (problem 2)
- Auxiliary role-prediction loss (problem 3)

---

## 2. Üst-Düzey Mimari

```
            ┌────────────────────────────────────────────┐
            │  TOKEN INPUT  ("jump twice")                │
            └─────────────────┬──────────────────────────┘
                              ▼
            ┌────────────────────────────────────────────┐
            │  FILLER ENCODER (Transformer)               │
            │    her token için d_filler boyutlu vektör   │
            │    f_i  ∈  R^{d_filler}                     │
            └─────────────────┬──────────────────────────┘
                              ▼
            ┌────────────────────────────────────────────┐
            │  ROLE INFERENCE (Hyperbolic head)           │
            │    her token için K rol üzerine             │
            │    Gumbel-softmax ⇒ r_i ∈ Poincaré^{d_role} │
            └─────────────────┬──────────────────────────┘
                              ▼
            ┌────────────────────────────────────────────┐
            │  TPR BINDING                                │
            │    T = Σ_i  f_i ⊗ r_i                       │
            │    T ∈ R^{d_filler × d_role}                │
            └─────────────────┬──────────────────────────┘
                              ▼
            ┌────────────────────────────────────────────┐
            │  REFINEMENT LOOP (K iter, adaptive halt)    │
            │    T_{k+1} = T_k + Attention(T_k, ctx)      │
            │    Halt when ||T_{k+1} - T_k|| < ε          │
            └─────────────────┬──────────────────────────┘
                              ▼
            ┌────────────────────────────────────────────┐
            │  TYPED DECODER (AR cross-attention)         │
            │    her step decoder query q üretir          │
            │    role activation: r̂ = softmax(R q)        │
            │    unbind: f̂ = T* · r̂                       │
            │    next token = LM_head(f̂, hidden)          │
            └─────────────────┬──────────────────────────┘
                              ▼
            ┌────────────────────────────────────────────┐
            │  TOKEN OUTPUT  ("I_JUMP I_JUMP")            │
            └────────────────────────────────────────────┘
```

---

## 3. Bileşen Tasarımı

### 3.1 Role Inventory

**Amaç:** Compositional rolleri (verb, modifier, direction, etc.) discrete ve isteğe bağlı hiyerarşik temsil etmek.

**Yapı:**
- K = 8 öğrenilebilir rol (Faz HC-1; daha sonra artırılabilir)
- Her rol d_role boyutlu embedding (default d_role=8, Euclidean)
- Rol embeddings öğrenilebilir, supervised role auxiliary loss ile yönlendirilir

**Geometri seçimi (hyperbolic flag):**
- **Default (Faz HC-1): Euclidean.** Stabil, basit, geoopt yok. d_role=8.
- **Opt-in (Faz HC-2 ablasyon): Hyperbolic.** Poincaré ball R^{d_role=5}. `tanh`-projected raw param ile manifold içinde tutulur. Riemannian Adam henüz kullanılmaz.

**Neden Euclidean-first:** LDC v2'de hyperbolic'i hiç denemedik; numerik instability + position-augmentation Möbius addition gerektirir. Önce Euclidean ile mimari hipotezi doğrula, sonra hyperbolic'in marginal kazanımını ölç (H4 ablasyon).

**Önemli karar:** Faz HC-1'de roller hand-anotasyon ile süpervize edilir (SCAN parser zaten var). Faz HC-3+'da unsupervised role induction denenir.

**Numerik notlar (hyperbolic mode için):**
- fp32 zorunlu, mixed precision kapalı
- `tanh` projection ile boundary'den 0.999 marjı
- Dim 5 minimum (2D unstable, LDC v2 Test 1)

---

### 3.2 Filler Encoder

**Amaç:** Her input token için "dolgu" vektörü üretmek.

**Yapı:**
- Standart transformer encoder, 2-4 layer, d_model = d_filler = 64-96
- Token embedding + positional encoding + multi-head self-attention
- LDC v2'deki encoder'ın sadeleştirilmiş hali (slot attention yok)

**Çıktı:** Her input pozisyonu için `f_i ∈ R^{d_filler}`.

**Ablasyon kancası:** Filler dimensyonu compositional gen'de etkili mi? (A_filler ablasyon)

---

### 3.3 Role Inference

**Amaç:** Her input token'a hangi rolün düştüğünü tahmin etmek.

**Yapı:**
- Filler vektör f_i'den K rol üzerine logit hesapla
- Gumbel-softmax (τ=0.5 → 0.1 schedule) ile differentiable discrete seçim
- Seçilen rol indeksi → Poincaré embedding lookup → r_i

**Pseudo:**
```python
logits = MLP(f_i)               # R^K
role_idx = gumbel_softmax(logits, tau)
r_i = exp_map_0(role_embed[role_idx])  # to Poincaré ball
```

**Auxiliary loss (Faz 1):**
- Eğitim sırasında "ground truth" rol etiketi varsa (SCAN: walk→action, twice→modifier)
- L_role = cross_entropy(logits, true_role)
- Toplam loss = α · L_lm + β · L_role

**Bu süpervizyon LDC v2'nin tam eksiği olan compositional sinyaldir.**

---

### 3.4 TPR Binding (Position-Augmented)

**Amaç:** Filler-role çiftlerini tek bir compositional tensor'a paketlemek **ve aynı role'a bind olan birden fazla token'ı disentangle etmek.**

**Naïve TPR'nin Sorunu (Multi-token Same Role):**

Standart TPR `T = Σ f_i ⊗ r_i` kullanıldığında, iki token aynı role bağlandığında (örn. "walk and jump" — her ikisi de action):
```
T = f_walk ⊗ r_action + f_jump ⊗ r_action
  = (f_walk + f_jump) ⊗ r_action
```
Unbinding `T · r_action` toplam vektörü geri verir. Walk veya jump tek tek geri alınamaz. Bilgi kaybı değil — **yıkıcı interferans**. SCAN'da "and" kombinasyonları bu yüzden patlar.

**Çözüm: Position-Augmented Binding**

Her token'a unique pozisyonel perturbation eklenmiş etkin role kullanılır:
```
r_eff_i = combine(r_role_i, p_i, α)
T = Σ_{i=1}^{N} f_i ⊗ r_eff_i
```

`p_i` öğrenilebilir position vector (d_role boyutlu, sequence position'a bağlı).
`α` küçük scalar coefficient (default 0.1, ablasyon-hedefli).

**Geometri-bağımlı `combine`:**
- **Euclidean (default):** `r_eff_i = r_role_i + α · p_i` — basit toplama.
- **Hyperbolic (opt-in):** `r_eff_i = mobius_add(r_role_i, α · p_i)` — Poincaré ball içinde kalmak için Möbius addition. (`p_i` log-space'te, exp-map ile manifold'a alınır.) Numerik karmaşıklık Euclidean'a kıyasla daha yüksek.

**α Trade-off:**
- α → 0: Naïve TPR; multi-role collision sorunu geri gelir.
- α → büyük: Pure role sharing erozyona uğrar; compositionality bozulur.
- **Sweet spot bulma:** Faz HC-0 toy task (3-4 token "action and action" örnekleri) ile α ∈ {0.01, 0.05, 0.1, 0.2, 0.5} sweep.

**Unbinding:**
```
f̂_for_role_at_position(r, p) = T · combine(r, p, α)
```
Decoder her step'te hem role query'si hem pozisyonel hint üretir; ikisinden `r_eff_query` türetilir.

**Boyut:**
- d_filler = 64-96
- d_role = 8 (Euclidean default) veya 5 (hyperbolic)
- T ∈ R^{B × d_filler × d_role}

**Position vector kaynağı:**
- Faz HC-1: Sinusoidal positional encoding (sequence index'ten türetilmiş)
- Faz HC-2 ablasyon: Learnable per-position embedding
- Edge case: Test-time'da training'den uzun sequence → sinusoidal extrapolation, learnable interpolation

**Alternatif binding mekanizmaları (Faz HC-3+ ablasyon olarak):**
- **Higher-order tensor:** `T = Σ f_i ⊗ r_i ⊗ p_i` (üçlü binding, decoder'da hem r hem p ile unbind)
- **Per-role dynamic slots:** Her role için ayrı slot attention; T tensor (B, d_filler, d_role, N_slot)
- **Relational binding:** Token'lar arası attention'la pairwise bound tensor

Bu alternatifler MVP kapsam dışı; ablasyon H8-H11 olarak değerlendirilir.

---

### 3.5 Refinement Loop

**Amaç:** Bound tensor T'yi iteratif olarak rafine etmek (LDC v2 diffusion'ın replacement'ı).

**Yapı:**
- T_0 = TPR binding çıktısı
- T_{k+1} = T_k + γ · Attention(T_k, encoder_context)
- Adaptive halting: ||T_{k+1} - T_k||_F < ε veya max_iter K=8

**Neden adaptive:** Basit cümleler tek iterasyonda çözülür, kompleks (nested) yapılar daha fazla iterasyon gerektirir. Halt-token (ACT, Universal Transformer) ile öğrenilir.

**LDC v2'den fark:** Diffusion yerine deterministic refinement. Noise yok. Hedef compositional yapıyı keskinleştirmek, semplemek değil.

**Eğitim sinyali:**
- Decoder her iterasyon T_k üzerinden çıkış üretir
- L_lm(T_K) yanı sıra her ara T_k'da supervisory loss da hesaplanabilir (deep supervision, isteğe bağlı)

---

### 3.6 Typed Decoder

**Amaç:** T*'den autoregressive olarak çıkış token'ları üretmek.

**Yapı:**
- Standart transformer decoder, 2-4 layer
- Self-attention + cross-attention
- **Cross-attention özel:** memory olarak T* tensor'unun reshape'i kullanılır
- Her decoder step:
  1. Self-attention output → query vektör q ∈ R^{d_model}
  2. Q'yu d_role'a project et: q_role
  3. Role activation: r̂ = softmax(role_embeddings @ q_role)
  4. Unbind: f̂ = T* · r̂ (hangi rolde hangi filler)
  5. f̂ ile self-attention output birleşip LM head'e gider

**Sıkı tip kısıtı (opsiyonel ablasyon):**
- Decoder her step'te sadece TEK rolü query'leyebilir (hard attention via Gumbel)
- Bu compositional bias'ı sertleştirir

---

## 4. Eğitim Protokolü

### 4.1 Loss

```
L_total = α_lm · L_lm + β_role · L_role + γ_cons · L_consistency
```

- **L_lm:** Standart cross-entropy decoder çıktısı vs target.
- **L_role:** Role inference auxiliary supervision. `cross_entropy(role_logits, true_role_labels)`. SCAN için role tagger (`src/data/role_tagger.py`) ground truth sağlar.
- **L_consistency:** Compositional invariance constraint.

**L_consistency formülü (explicit):**

Aynı yapısal şablon (örn. `<action> twice`) farklı filler'larla iki örnek üretir. Bound tensor T'lerinin role-projected özellikleri benzer olmalı:
```
For batch pair (x_a, x_b) with same role-sequence pattern:
  T_a = bind(filler_a, role_a)
  T_b = bind(filler_b, role_b)
  proj_a = T_a · r_template  (e.g. r_modifier)
  proj_b = T_b · r_template
  L_cons = MSE(normalize(proj_a), normalize(proj_b))
```
Pattern matching: source token role sequence eşitse same-template kabul edilir. Negatif pairs (farklı pattern) için contrastive margin loss da denenebilir (Faz HC-3 ablasyon).

**Faz HC-1 başlangıç ağırlıkları:** α_lm=1.0, β_role=0.5, γ_cons=0.1.

### 4.2 Curriculum

```
Phase 1.0: Role-only pretraining        (5K step; sadece L_role, decoder dondur)
Phase 1.1: Atomic primitives only       (walk → I_WALK)
Phase 1.2: Simple compositions          (walk twice → I_WALK I_WALK)
Phase 1.3: Nested compositions          (walk twice and jump → ...)
Phase 1.4: Full SCAN train data
```

**Phase 1.0 (yeni):** Role inference head ve role inventory'i izole eğit. Decoder + refinement dondurulur. Bu adım L_role'u stabilize eder, sonraki phase'lerde joint training daha temiz olur.

**Phase 1.1-1.4:** Joint training. Her sub-phase 5K step. Toplam 25K step (Phase 1.0 dahil).

**Curriculum hipotezi:** Roller önce basit ortamda öğrenilirse sonradan kompoze edilmesi kolaylaşır.

### 4.3 Optimizer

- AdamW (default; Faz HC-1 Euclidean modunda yeterli)
- Riemannian Adam: yalnızca hyperbolic flag açıkken role inventory için (Faz HC-2+)
- LR warmup 1K step, cosine decay
- Mixed precision: role pathway dışında AMP açık (Euclidean modda problem yok)

---

## 5. Değerlendirme

### 5.1 Birincil Benchmark — SCAN

LDC v2 baseline'ı + LDC v2 ile aynı 3 split:
- `simple` — kontrol (her ikisi de %100 vermeli)
- `addprim_jump` — ana hipotez testi
- `template_around_right` — generalisation testi

**Başarı kriterleri:**
- `addprim_jump` test seq_acc ≥ %20 → mimari hipotez canlı
- ≥ %50 → güçlü compositional bias
- ≥ %80 → state-of-the-art seviyesi (literatürde özel modeller)

### 5.2 İkincil — COGS

`addprim_jump`'tan sonra COGS benchmark'ında test (daha rigoröz compositional gen).

### 5.3 Ablasyonlar

| Varyant | Değişiklik | Hipotez |
|---------|-----------|---------|
| H0 (baseline) | Vanilla transformer | Referans |
| H1 (LDC v2) | LDC v2 (mevcut) | Tipsiz slot baseline |
| H2 (HiCoRe full) | Tüm bileşenler | Ana model |
| H3: no role sup | β=0 (L_role kapalı) | Role auxiliary loss kritik mi? |
| H4: flat roles | Öklid roller, hyperbolic kapalı | Hyperbolic gerekli mi? |
| H5: no refinement | K=1 (tek iterasyon) | Iterative refinement gerekli mi? |
| H6: no curriculum | Random data sırası | Curriculum gerekli mi? |
| H7: no TPR | Slot attention (LDC v2 gibi) | TPR binding gerekli mi? |

Her ablasyon en az 2 seed.

---

## 6. Risk Kaydı

| ID | Risk | Olasılık | Hafifletme |
|----|------|----------|------------|
| R1 | Role inference unsupervised çalışmaz | Yüksek | Faz 1'de supervised, sonra unsupervised dene |
| R2 | TPR boyutları patlar (d_filler × d_role) | Orta | Düşük d_role (5-32), low-rank approx |
| R3 | Hyperbolic numerik instability | Orta | fp32, projection clipping (LDC v2'den ders) |
| R4 | Adaptive halting öğrenilmez | Orta | İlk başlangıçta sabit K=4, sonra halting eklenir |
| R5 | Curriculum overfitting (Phase 1.1'de tıkanma) | Düşük | Her sub-phase'te global validation |
| R6 | LDC v2 gibi val %100 ama test %0 | Yüksek | Held-out test'i her eval'de ölç (LDC v2 dersi) |

---

## 7. Implementasyon Yol Haritası

### Faz HC-0: Hazırlık (1 hafta)
- `src/hicore/` modül iskeleti
- Birim testler:
  - `test_role_inventory.py` (Poincaré projeksiyon, gradient flow)
  - `test_tpr_binding.py` (binding/unbinding sayısal sağlık)
  - `test_refinement_loop.py` (halting gradient flow)

### Faz HC-1: Çekirdek (1-2 hafta)
- Filler encoder (LDC v2 encoder'dan port)
- Role inference head (Gumbel-softmax + Poincaré)
- TPR binding modülü
- Iterative refinement (sabit K, halting yok)
- Typed decoder
- End-to-end smoke test (1-batch overfit)

### Faz HC-2: Eğitim + İlk Sonuç (1 hafta)
- SCAN data loader yeniden kullan
- Curriculum scheduler
- Auxiliary role loss
- 3 split full eğitim, held-out test eval
- Ablation H7 (no TPR) ek run

### Faz HC-3: Ablasyonlar (1-2 hafta)
- H3-H6 ablasyonları
- Sonuç tablosu, hangi bileşen ne kadar katkı
- `results/2026-XX-XX-hicore-ablations.md` raporu

### Faz HC-4: Adaptive Halting (1 hafta)
- ACT-tarz halting mekanizması
- Histogram analizi: zor örnekler daha çok iter mi kullanıyor?

### Faz HC-5: COGS + Genişletilmiş Benchmark (2 hafta)
- COGS dataset entegrasyonu
- gSCAN (varsa)
- Karşılaştırmalı sonuçlar

### Faz HC-6: Karar Gate'i (1 hafta)
- Tüm sonuçlar
- LDC v2 ile karşılaştırma tablosu
- Devam/pivot/rapor kararı

---

## 8. Kaynak Tahmini

| Kaynak | Min | Hedef |
|--------|-----|-------|
| GPU | Lightning AI Tesla T4 (mevcut) | T4 yeterli, A100 lüks |
| Kredi | ~5-10 saat (Faz HC-1+HC-2) | 30-50 saat (tam ablasyon) |
| Wall-clock | 1 ay (part-time) | 6 hafta nominal |
| Disk | 2 GB ek (checkpoints + logs) | 10 GB |
| Yeni dependency | `geoopt` (hyperbolic ops) | — |

---

## 9. Mevcut Kod Tabanı ile İlişki

**Yeniden kullanılacak (LDC v2'den):**
- `src/data/scan.py` (dataset loader, tokenizer)
- `src/data/tokenizer.py` (WordTokenizer)
- `src/baselines/transformer.py` (baseline, kontrol)
- `src/train.py` (training loop, eval, test_loader pipeline)
- `src/eval.py` (held-out eval)
- Hydra config sistemi

**Yeni yazılacak:**
- `src/hicore/role_inventory.py` (Poincaré rol embed)
- `src/hicore/role_inference.py` (Gumbel-softmax head)
- `src/hicore/tpr.py` (binding/unbinding)
- `src/hicore/refinement.py` (iterative loop, halting)
- `src/hicore/decoder.py` (typed cross-attention)
- `src/hicore/model.py` (uçtan uca HiCoReModel)
- `experiments/configs/hicore_*.yaml` (3 SCAN split + ablasyonlar)

**Olası refactor:**
- `src/modules/` ortak (positional, masking) — mevcut kullanılır
- LDC v2 modülleri silinmez, ablasyon karşılaştırması için kalır

---

## 10. Açık Sorular (Faz HC-0'da Kararlaştırılacak)

1. **Rol envanteri büyüklüğü K:** 8 yeterli mi yoksa 16/32 mi? (SCAN için 8 makul, COGS belirsiz)
2. **d_role:** Poincaré 5D (literatür minimum) yeterli mi?
3. **L_role supervision kaynağı:** SCAN'da grammar var; COGS'ta otomatik tagger gerek mi?
4. **Refinement halting:** Hard halt (single break) mi soft (weighted sum) mu?
5. **Multi-token same-role:** İki action varsa T'de ayrım nasıl? Pozisyonel embedding role'a eklensin mi?
6. **Inference time:** Refinement K iter, decoder N step. Toplam baseline'a göre kaç kat yavaş?

Faz HC-0 sonunda bu soruların hepsi cevaplı olmalı.

---

## 11. Bilimsel Katkı İddiası

HiCoRe **yeni** bir mimari değil — bileşenleri (TPR, hyperbolic embeds, iterative refinement) literatürde mevcut. Yeni olan:

1. **Bu üçünün ilk kez compositional gen problemi için birleşimi**
2. **Hierarchical hyperbolic role inventory** (flat TPR roller'ın evrimi)
3. **Adaptive halting + TPR refinement birleşimi** (sabit-iter Universal Transformer'ın evrimi)
4. **LDC v2 başarısızlığından çıkarılan tasarım kararları** (auxiliary role loss, supervised role learning Faz 1'de)

Hedef yayın seviyesi: NeurIPS / ICLR workshop (gerçekçi), main track (sonuçlar güçlü çıkarsa).

---

## 12. Son Söz

LDC v2 hipotezi (slot attention + diffusion ⇒ compositional gen) yanlış çıktı. Bu bilgi değerli; literatüre eksikti.

HiCoRe yeni bir hipotez: **compositional gen, mimaride explicit binding ve typed roles ile kazanılır.** Bu hipotez 30 yıllık symbolic AI iddialarının modern neural reincarnation'ıdır.

Test ediyoruz. Yanlış çıkarsa, o da değerli bilgi.
