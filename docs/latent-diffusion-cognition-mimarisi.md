# Latent Diffusion Cognition (LDC)
## Anlam Uzayında Düşünen Yeni Nesil Yapay Zeka Mimarisi

> *"Düşünce kelimelerden önce gelir. Mevcut dil modelleri bu adımı atlıyor."*

---

## 1. Vizyon ve Motivasyon

### 1.1 Mevcut Paradigmanın Sınırları

Bugünkü büyük dil modelleri (GPT, Claude, Gemini vb.) **otoregresif token tahmini** üzerine kurulu. Yani metni soldan sağa, kelime kelime üretiyorlar. Bu yaklaşımın üç temel zayıflığı var:

1. **Geri dönüşsüzlük:** Bir token üretildikten sonra değiştirilemez. Model erken bir hata yaparsa, üzerine inşa etmek zorundadır.
2. **Sabit hesap bütçesi:** "Merhaba" demek için harcanan hesap, "Riemann hipotezini açıkla" demek için harcananla aynıdır.
3. **Planlama yokluğu:** Model bir cümleye başlarken nereye gideceğini bilmez. Her token, o ana kadar olan koşullu olasılıktır.

### 1.2 Temel Hipotez

İnsan zihni böyle çalışmaz. Bir fikri ifade etmeden önce **ne demek istediğimizi biliriz**, sonra dile dökeriz. Düşünce dilden önce gelir.

LDC bu sezgiyi mimariye dönüştürür:

> Modelin çıktısını token uzayında değil, **anlam uzayında** üretmesi; bu temsili **iteratif olarak rafine etmesi**; ve **en son aşamada** dile çevirmesi.

---

## 2. Mimari Genel Bakış

LDC üç ana modülden oluşur:

```
[Girdi: Doğal dil]
        ↓
┌─────────────────────────────┐
│  1. ENCODER                 │  Dilden anlam uzayına
│  (Language → Concept Space) │
└─────────────────────────────┘
        ↓
   [Concept Configuration C₀]
        ↓
┌─────────────────────────────┐
│  2. LATENT DIFFUSION CORE   │  İteratif anlamsal rafinasyon
│  (Iterative Refinement)     │
└─────────────────────────────┘
        ↓
   [Refined Configuration C*]
        ↓
┌─────────────────────────────┐
│  3. DECODER                 │  Anlamdan dile
│  (Concept Space → Language) │
└─────────────────────────────┘
        ↓
[Çıktı: Doğal dil]
```

Her modül kendi içinde alt sistemlere ayrılır. Önemli olan şudur: **çekirdek hesaplama dilde değil, anlam uzayında olur.**

---

## 3. Concept Space (Anlam Uzayı)

### 3.1 Geometrik Yapı

Anlam uzayı düz Öklid uzayı **değildir**. Çünkü kavramsal ilişkiler (hiyerarşi, kategorilik, soyutluk seviyeleri) Öklid geometrisinde doğal yer bulmaz.

Önerilen yapı: **Hibrit manifold**.

- **Hiperbolik bileşen** (Poincaré disk modeli): Hiyerarşik ilişkiler için. Hiperbolik uzayda hacim üstel olarak büyür — bu, ağaç yapılarını minimum bozulmayla gömmenize olanak verir. "Canlı → memeli → köpek → golden retriever" gibi taksonomiler doğal olarak yerleşir.
- **Öklid bileşen:** Sürekli özellikler için (renk, boyut, zaman, duygu yoğunluğu vb.).
- **Toroidal bileşen:** Döngüsel kavramlar için (yıl, gün, mevsim, müzik tonu).

Bir kavram, bu üç manifoldun çarpım uzayında bir nokta olarak temsil edilir:

```
c ∈ ℋⁿ × ℝᵐ × 𝕋ᵏ
```

burada `ℋⁿ` hiperbolik, `ℝᵐ` Öklid, `𝕋ᵏ` toroidal manifoldlardır.

### 3.2 İlişkiler Birinci Sınıf Nesnelerdir

Klasik embedding yaklaşımında ilişkiler vektör farklarıyla ifade edilir (ünlü "king - man + woman = queen" örneği). Bu kabadır.

LDC'de **ilişkilerin kendileri de uzayın elemanlarıdır**. Bir kavram konfigürasyonu yalnızca noktalardan değil, noktalar arası tipli kenarlardan oluşan bir **graf**'tır:

```
C = (V, E)
V = { kavram noktaları } ⊂ ℋⁿ × ℝᵐ × 𝕊ᵏ
E = { (vᵢ, vⱼ, rᵢⱼ) | rᵢⱼ ∈ R } 
R = { ilişki tipleri: nedensel, zamansal, hiyerarşik, ... }
```

Bu, kategori teorisindeki **morphism** kavramına yakındır. Modelin "düşüncesi" sadece nesneler değil, nesneler arası dönüşümlerdir.

### 3.3 Belirsizlik Modellemesi

Her kavram noktası deterministik değil, **olasılıksal** bir bulutdur. Yani her kavram, manifold üzerinde bir dağılımdır:

```
c ~ p(c) defined on M
```

Bu, modelin "tam karar vermedim" diyebilmesini sağlar. Belirsizlik diffusion sürecinin temelidir.

---

## 4. Diffusion Core (Çekirdek)

### 4.1 Forward Process — Anlamsal Gürültü

Klasik image diffusion'da gürültü Gaussian'dır. Bizde değil. Çünkü "anlam" üzerine eklenmiş Gaussian gürültü mantıklı bir yorum vermez.

Üç tip semantik gürültü tanımlarız:

#### Tip 1: Konsept Bulanıklaştırma
Bir kavram noktası, manifold üzerinde komşularına doğru "yayılır". Gürültü seviyesi `t` arttıkça, "köpek" → "köpek-kurt-tilki karışımı" → "memeli" → "canlı" şeklinde abstraksiyon merdiveninde yukarı çıkar.

```
Forward(c, t) = projeksiyon(c, abstraksiyon_seviyesi(t))
```

#### Tip 2: İlişki Belirsizleştirme
Graf kenarlarının tipleri olasılıksal hale gelir. "A, B'nin nedenidir" → "A ve B ilişkili, yön belirsiz" → "A ve B arasında bir ilişki var" → "A ve B".

#### Tip 3: Topolojik Pertürbasyon
Düşük gürültüde graf yapısı korunur, yüksek gürültüde kenarlar rastgele eklenir/silinir. Tam gürültüde sadece izole, tipsiz noktalar kalır.

Bu üç tipin birleşimi `t ∈ [0, T]` boyunca uygulanır.

### 4.2 Reverse Process — Düşünce İnşası

Modelin asıl işi: tam gürültüden başlayarak (rastgele bir konsept bulutu), adım adım anlamlı bir konfigürasyon inşa etmek.

```
C_T (gürültü) → C_{T-1} → ... → C_1 → C_0 (rafine düşünce)
```

Her adımda model şu soruyu sorar:
> "Bu konfigürasyondaki belirsizlik, bağlam göz önüne alındığında nasıl azaltılmalı?"

Bağlam: girdideki encoded mesaj + buraya kadar oluşmuş konfigürasyon.

### 4.3 Adaptif Hesaplama (Adaptive Computation)

Kritik yenilik: **adım sayısı sabit değildir.** Model her noktada şu kararı verir:

> "Konfigürasyon yeterince rafine mi? Durayım mı, devam edeyim mi?"

Bu bir **halt token**'ı veya bir **uncertainty threshold** üzerinden yapılır:

```
if uncertainty(C_t) < τ_task:
    return C_t
else:
    continue refinement
```

Basit problemler 3-5 adımda biter. Karmaşık akıl yürütme 50-100 adım sürebilir. Bu **insan düşüncesinin doğal ritmine** uyar — "Merhaba" demek anlık, bir matematik problemi düşünmek dakikalar sürer.

### 4.4 Mimari Detay: Refinement Network

Reverse step'i öğrenen ağ bir **graf transformer**'dır:

- Düğümler: kavram noktaları
- Kenarlar: tipli ilişkiler
- Attention: kenar tipine duyarlı, manifold-aware mesafe metriği kullanan

```
attention(vᵢ, vⱼ) = f(d_M(vᵢ, vⱼ), type(eᵢⱼ), context)
```

burada `d_M` hibrit manifoldun metriğidir (hiperbolik, Öklid, toroidal bileşenlerin uygun ağırlıklı kombinasyonu).

---

## 5. Encoder ve Decoder

### 5.1 Encoder: Dil → Anlam

Girdi metni, anlam uzayında bir başlangıç konfigürasyonuna dönüştürülür. Bu konfigürasyon **eksiktir** — sadece "ne sorulduğunu" temsil eder, "cevabı" değil.

Encoder transformer-tabanlıdır ama çıktısı tokenler değil, **(düğüm, kenar) çiftleridir**. Yani metnin semantik graf temsilini çıkarır.

### 5.2 Decoder: Anlam → Dil

Burası ilginçleşiyor. Decoder de **kendi içinde diffusion** kullanır, ama bu sefer dilde.

Latent konfigürasyon `C*` verilmişken, decoder cümle yapısını → kelime seçimini → ince ayarı sırasıyla rafine eder. Bu da iteratiftir, ama daha az adımla (genelde 5-10).

Bu çift-katmanlı diffusion (anlamda + dilde) insan dil üretimine yakındır:
1. Ne demek istiyorum? (anlam diffusion'u)
2. Bunu nasıl söylesem? (dil diffusion'u)

---

## 6. Eğitim Stratejisi

### 6.1 Aşamalı Eğitim

**Aşama 1: Encoder/Decoder pre-training.**
Büyük metin korpusunda, encoder-decoder'ı kimlik fonksiyonu olarak eğit. Yani metni anlam uzayına çevir, sonra geri dile çevir, ve original metnin yeniden inşasını maksimize et.

**Aşama 2: Latent diffusion training.**
Encoder'ı dondur. Concept space'te diffusion modelini standart denoising loss ile eğit:

```
L_diff = E[||ε_θ(C_t, t, context) - ε||²]
```

burada `ε` semantik gürültüdür ve `ε_θ` modelin tahmin ettiği gürültüdür.

**Aşama 3: End-to-end fine-tuning.**
Tüm sistemi birlikte ince ayarla. Reinforcement learning veya doğrudan supervised loss kullanılabilir.

**Aşama 4: Adaptive computation training.**
"Ne zaman duracağını" öğrenmesi için ayrı bir loss terimi:

```
L_total = L_task + λ · n_steps
```

Hem doğru cevabı bulmasını hem de minimum adımda bulmasını teşvik eder.

### 6.2 Veri Gereksinimleri

LDC standart LLM'lere göre **muhtemelen daha az veriye ihtiyaç duyacaktır**, çünkü kompozisyonelliği mimari olarak yakaladığı için sample efficiency yüksek olmalı. Ama bu hipotez — deneysel olarak doğrulanmalı.

Ek olarak: anlam uzayının kalitesi için **graf-yapılı veriler** (knowledge graph'lar, AMR parse'lar, sembolik matematiksel ifadeler) kritik olacaktır.

---

## 7. Doğrulama Metodolojisi

### 7.1 Aşama 1: Toy Problemler

Mevcut LLM'lerin başarısız olduğu, mimari avantajın görünmesi gereken görevler:

- **SCAN, COGS:** Kompozisyonel genelleme. Eğitimde "kırmızı top" ve "büyük kutu" gör, testte "büyük kırmızı top" doğru üret.
- **Blocksworld, Tower of Hanoi:** Çok adımlı planlama.
- **Self-correction tasks:** "Bu problemi çöz, sonra çözümünü doğrula ve gerekirse düzelt."
- **Long-context coherence:** 10k+ token'lık bir hikayede tutarlılık.

### 7.2 Aşama 2: Scaling Laws

LDC'nin scaling eğrisi transformer baseline'ından **daha dik** mi? Aynı parametre/veri/hesap bütçesinde loss daha hızlı mı düşüyor?

### 7.3 Aşama 3: Mekanistik Doğrulama

Latent uzayda **gerçekten** kavramlar oluşuyor mu? Probing classifier'lar ve sparse autoencoder'lar ile concept space'in iç yapısı incelenmeli. Hipotez ettiğimiz hibrit geometri ortaya çıkıyor mu?

### 7.4 Aşama 4: Önceden Tahmin

Bir teorinin gücü, önceden tahminler yapabilmesidir. Yayınlamadan önce şunlar tahmin edilmeli:
- LDC hangi görevlerde transformer'ı geçecek?
- Hangi görevlerde geride kalacak ve **neden**?
- Adaptive computation hangi soru tiplerinde otomatik olarak daha fazla adım kullanacak?

Tahminler doğru çıkarsa, sadece bir model değil, bir **teori** elimizdedir.

---

## 8. Mevcut Sistemlerle Karşılaştırma

| Özellik | Otoregresif LLM | Diffusion LM (Mercury vb.) | LDC |
|---------|-----------------|---------------------------|-----|
| Üretim yönü | Soldan sağa | Paralel, dilde | Paralel, anlamda |
| Geri düzeltme | ❌ | ✓ (token seviyesi) | ✓ (anlam seviyesi) |
| Adaptif hesap | Sınırlı (CoT ile) | Sabit adım | ✓ Doğal |
| Kompozisyonellik | Zayıf | Orta | Güçlü (mimari) |
| Yorumlanabilirlik | Düşük | Düşük | Yüksek (graf yapısı) |
| Hesap maliyeti | O(n²) attention | Çoklu adım | Çoklu adım + graf op. |
| Olgunluk | Çok yüksek | Düşük | Yok (önerilen) |

---

## 9. Açık Sorunlar ve Riskler

LDC'nin başarısı için çözülmesi gereken zor problemler var:

**P1: Anlam uzayının doğru parametreleşmesi.**
Hiperbolik+Öklid+toroidal hibrit kararı bir hipotez. Doğru geometri farklı olabilir. Belki öğrenilebilir bir manifold gerekir.

**P2: Sürekli ve ayrık arasındaki köprü.**
Anlam uzayı sürekli, ama "bu kavram tam olarak şudur" gibi ayrık kararlar gerekir. Gumbel-softmax, vector quantization gibi teknikler — ama hiçbiri tam çözüm değil.

**P3: Hesaplama maliyeti.**
Diffusion + graf operasyonları + adaptive computation. Naive uygulama transformer'dan çok yavaş olabilir. Donanım optimizasyonu ciddi araştırma gerektirir.

**P4: Eğitim kararsızlığı.**
Çok bileşenli sistemler eğitilmesi zordur. Encoder, diffusion core, decoder her biri farklı sinyallere optimize olur.

**P5: Değerlendirme zorluğu.**
"Anlam uzayında doğru düşündü mü" sorusu, "doğru çıktı verdi mi"den daha zor bir sorudur. Yorumlanabilirlik araçları bu mimariyle birlikte geliştirilmelidir.

---

## 10. Yol Haritası

**Faz 1 (0-6 ay): Kavram kanıtı.**
Küçük ölçekte (10M-100M parametre), oyuncak problemlerde LDC'nin temel mekanizmalarını çalıştır. Concept space'in basit bir versiyonu (sadece Öklid). Toy diffusion. Hedef: kompozisyonel genellemede transformer baseline'ını geçmek.

**Faz 2 (6-18 ay): Geometrik zenginleştirme.**
Hibrit manifoldu uygula. Graf-tabanlı concept representation. Adaptive computation. 1B parametre ölçeği.

**Faz 3 (18-36 ay): Ölçek.**
10B+ parametre. Gerçek dünya görevleri. Scaling law analizi. Açık sorulara empirik cevaplar.

**Faz 4 (36+ ay): Hibrit sistemler.**
LDC ile otoregresif modelleri birleştiren meta-mimari. "Ne zaman hangi modu kullanmalı" kararı verebilen bir sistem. Çünkü insan zihni de tek bir modda çalışmaz.

---

## 11. Felsefi Not

LDC sadece bir mühendislik önerisi değil, bir **bilişsel hipotez**dir.

Hipotez şudur: *Düşünce, dilden ayrı ve dilden önce var olan, kendi geometrisi olan bir şeydir. Yapay zekayı insan-seviyesi anlamaya yaklaştırmak istiyorsak, modeli bu seviyede çalıştırmalıyız.*

Bu hipotez yanlış olabilir. Belki düşünce gerçekten dilseldir ve "anlam uzayı" sadece bir soyutlamadır. O zaman LDC daha iyi sonuç vermez.

Ama doğruysa — eğer gerçekten kavramsal düşünce dilden bağımsız bir yapıysa — o zaman LDC, mevcut paradigma transformer'ın yapamayacağı bir şeyi yapacaktır: **gerçekten düşünmek**, sadece akıcı konuşmak değil.

Bu denemeye değer bir bahis.

---

## Ek: Terim Sözlüğü

- **Concept Space (Anlam Uzayı):** Kavramların ve aralarındaki ilişkilerin yaşadığı geometrik yapı.
- **Latent Configuration:** Concept space'teki bir nokta-kenar grafının somut hali; modelin "düşüncesi".
- **Semantic Noise:** Anlam uzayında tanımlanmış, kavramsal bulanıklaştırma anlamına gelen gürültü tipi.
- **Adaptive Computation:** Modelin görev zorluğuna göre kendi hesap bütçesini belirlemesi.
- **Hibrit Manifold:** Hiperbolik + Öklid + toroidal manifoldların çarpımı.
- **Refinement Network:** Diffusion'ın reverse adımını öğrenen graf transformer.

---

*Bu doküman, gerçek bir araştırma programının başlangıcı için bir taslaktır. Her bölüm aylar süren araştırma ve deneysel doğrulama gerektirir. Mimari mükemmel değil — ama farklı, ve denenmeye değer.*
