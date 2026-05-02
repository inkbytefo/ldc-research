# LDC'yi Geliştirme Rehberi
## Sıfırdan Kavram Kanıtına Pratik Yol Haritası

---

## Önemli Uyarı

Bu rehber 12-24 aylık ciddi bir taahhüt varsayar. Tam zamanlı çalışırsan 12 ay, hafta sonu/akşamları çalışırsan 24+ ay. Daha kısa süre düşünüyorsan kapsamı küçültmen gerekir.

Hedef: **toy problemlerde transformer baseline'ını yenen küçük ölçekli bir LDC prototipi**. Tam ölçekli sistem değil.

---

## Bölüm A: Kişisel Hazırlık (0-3 Ay)

Bu bölüm atlanırsa proje çuvallar. Çoğu kişinin atladığı kısım.

### A1. Ön gereksinim bilgi

Şunları bilmen lazım, en azından çalışma seviyesinde:

**Matematik:**
- Lineer cebir (matris çarpımı, eigen-decomposition, SVD)
- Çok değişkenli kalkülüs (gradient, jacobian, chain rule)
- Olasılık (koşullu olasılık, KL divergence, expectations)
- Temel diferansiyel geometri (manifold kavramı, tangent space) — bu çoğu ML kişisinde yok, senin avantajın olabilir

**Programlama:**
- Python (akıcı seviye)
- PyTorch (en az 6 ay deneyim)
- Git, command line, Linux

**ML temelleri:**
- Backpropagation'ın nasıl çalıştığını **gerçekten** anlamak
- Transformer mimarisini sıfırdan implement edebilmek (Karpathy'nin nanoGPT'sini takip et)
- Diffusion modellerinin matematiğini bilmek (DDPM paper'ı + birkaç implementasyon)

Eksiklerini şu sırayla kapat:
1. Karpathy'nin "Neural Networks: Zero to Hero" YouTube serisi (ücretsiz, paha biçilmez)
2. "The Annotated Diffusion Model" (Hugging Face blog)
3. Nickel & Kiela'nın "Poincaré Embeddings" makalesi (hiperbolik geometri için)

### A2. Donanım

Bu projede en az şu donanıma erişimin olmalı:

**Minimum:** Tek GPU, 16GB+ VRAM. RTX 3090, 4090, ya da bulutta A100. Kendi makinende yoksa:
- Google Colab Pro+ (~50$/ay): A100 erişimi
- RunPod, Lambda Labs (saatlik kiralama): A100 ~1-2$/saat
- Vast.ai: daha ucuz ama daha az güvenilir

**İdeal:** 2-4 GPU node. Aylık ~500-2000$ bütçe.

Bütçen yoksa: küçükten başla. İlk 6 ayda Colab Free + Colab Pro yeterli olabilir.

### A3. Ortam kurulumu

```bash
# Python 3.11 + Conda environment
conda create -n ldc python=3.11
conda activate ldc

# Temel bağımlılıklar
pip install torch torchvision  # CUDA versiyonu sistemine göre
pip install numpy scipy matplotlib
pip install wandb  # deney takibi için kritik
pip install hydra-core  # config yönetimi
pip install einops  # tensor operasyonları
pip install geoopt  # hiperbolik geometri için

# Geliştirme araçları
pip install jupyter ipython
pip install black ruff mypy  # kod kalitesi
pip install pytest  # test
```

GitHub repo aç, hesabını wandb.ai'a bağla. Her deneyi wandb'ye logla — yoksa 3 ay sonra hangi denemenin ne yaptığını hatırlamazsın.

---

## Bölüm B: Faz 1 — En Küçük Çalışan Versiyon (3-6 Ay)

Hedef: **çalışan ama çok basitleştirilmiş bir LDC**. Tüm radikal fikirleri unut, en basit hali kur.

### B1. Tasarım sadeleştirmeleri (Faz 1 için)

Asıl tasarımdan şu basitleştirmeleri yap:

| Asıl LDC | Faz 1 versiyonu |
|----------|-----------------|
| Hibrit manifold (hiperbolik + Öklid + toroidal) | Sadece Öklid |
| Graf-yapılı concept space | Sabit boyutlu vektör (sıralı slot'lar) |
| 3 tip semantik gürültü | Sadece Gaussian |
| Adaptive computation | Sabit adım sayısı (örn. 20) |
| Çift katmanlı diffusion | Sadece latent diffusion, decoder basit |

Yani Faz 1, aslında **"latent diffusion + transformer encoder/decoder"** sandwich'idir. LDC'nin ruhu var ama tüm karmaşık fikirleri sonraya bırakıyoruz.

Bu önemli: **önce çalışan bir şey, sonra iyileştirme**. Tüm fikirleri aynı anda denemek başarısızlık reçetesidir.

### B2. Veri seti seçimi

İlk testler için **SCAN** datasetiyle başla:
- https://github.com/brendenlake/SCAN
- Küçük (~20k örnek)
- Kompozisyonel genelleme test eder
- Transformer'ların bilinçli zayıf olduğu yer
- Çalışmak kolay

İkinci olarak **gSCAN** veya **COGS** ekle — daha zorlu kompozisyonel testler.

### B3. Baseline kurulumu

Önce LDC'yi değil, **karşılaştıracağın transformer baseline**'ını implement et. Bu ~1 hafta sürer ama atlama:

1. Küçük bir transformer (6 katman, 256 hidden dim) yaz
2. SCAN üzerinde eğit
3. Test setinde başarısını ölç
4. Sonuçları wandb'ye logla

Bu rakamı hatırla. LDC'nin geçmesi gereken çizgi bu.

### B4. LDC iskeleti

Şimdi LDC'yi yaz. Modüler yapı:

```python
# ldc/
├── encoder.py          # Metin → latent vector
├── diffusion_core.py   # Latent diffusion (DDPM-style)
├── decoder.py          # Latent → metin
├── model.py            # Hepsini birleştiren ana modül
├── train.py            # Eğitim döngüsü
├── eval.py             # Değerlendirme
└── configs/            # Hydra config dosyaları
```

**Encoder:** Standart transformer encoder. Çıktı: tek bir latent vektör (mean pooling). Boyut: 256.

**Diffusion Core:** En basit hali. Latent vektör üzerinde DDPM. Conditioning: encoded girdi.

```python
# Pseudo-code
def diffusion_step(z_t, t, context):
    # z_t: latent at time t
    # t: timestep
    # context: encoded input
    eps_pred = denoising_network(z_t, t, context)
    z_t_minus_1 = ddpm_step(z_t, eps_pred, t)
    return z_t_minus_1
```

**Decoder:** Conditional transformer decoder. Latent vektörü cross-attention üzerinden alır, metin üretir.

### B5. Eğitim stratejisi

Önerdiğim sıra:

1. Encoder + Decoder'ı birlikte autoencoder olarak eğit (latent diffusion olmadan). 1 hafta. Eğer girdiyi reconstruct edebiliyorsa, latent space mantıklı şekilde organize olmuştur.

2. Encoder ve decoder'ı dondur. Latent diffusion'ı eğit. 1-2 hafta. Loss DDPM standart loss'u.

3. Tüm sistemi end-to-end fine-tune et. Birkaç gün.

Her aşamada SCAN test setinde değerlendir. Wandb'de loss eğrilerini izle.

### B6. Faz 1 başarı kriteri

LDC, transformer baseline'ını **kompozisyonel test split**'inde geçmeli. SCAN'da bunlar:
- "add jump" split (~%10 baseline accuracy genelde)
- "around right" split

Eğer baseline %10 alıyorsa, LDC %20+ almalı. Bu küçük gibi görünür ama mimari avantajın gerçek kanıtıdır.

**Geçemezse:** Mimari hipotez şu ölçekte çalışmıyor demektir. Ya basitleştirmeleri farklı yap, ya da temel varsayımları sorgula. Devam etmeden bunu çöz.

---

## Bölüm C: Faz 2 — Geometrik Zenginleştirme (6-12 Ay)

Faz 1 çalıştıysa, asıl LDC fikirlerini eklemeye başla. **Tek seferde değil, birer birer.** Her ekleme bir kontrol deneyi.

### C1. Hiperbolik bileşen ekle

Latent uzayın bir kısmını hiperbolik yap. `geoopt` kütüphanesini kullan:

```python
import geoopt
hyperbolic_manifold = geoopt.PoincareBall(c=1.0)
# Latent vector: yarısı Öklid, yarısı hiperbolik
```

Optimizer'ı buna göre ayarla (Riemannian Adam). Bu zor, ama geoopt yardımcı olur.

Test: hiyerarşik kavramları içeren bir görevde (WordNet alt-kümesi gibi) iyileşme var mı?

### C2. Graf yapılı concept

Tek vektör yerine, latent uzayda birden fazla "slot" tut. Her slot bir kavram. Slotlar arası ilişkiler attention ile öğrenilsin.

Bu Slot Attention paper'ında benzer şekilde yapılmış. Onu referans al.

### C3. Adaptive computation

Sabit 20 adım yerine, modelin "yeterince düşündüm" demesine izin ver:

```python
def adaptive_diffusion(z_T, context, max_steps=50):
    z = z_T
    for t in range(max_steps, 0, -1):
        z = diffusion_step(z, t, context)
        confidence = predict_confidence(z, context)
        if confidence > threshold:
            break
    return z
```

`predict_confidence` ayrı bir küçük ağdır, eğitim sırasında öğrenir.

Test: kolay vs zor problemlerde adım sayısı farkı oluşuyor mu?

### C4. Semantic noise

En son yap, çünkü en zor. İlk versiyonu basit tut: Gaussian gürültü yerine, "kavram komşusuna doğru sapma". Manifold üzerinde tanjant uzayda Gaussian sample et, exponential map ile manifolda gönder.

---

## Bölüm D: Yazma ve Yayınlama (12-18 Ay)

Mimariyi geliştirmek bir şey, **dünyanın bunu görmesi** ayrı şey. Bu fazı atlama.

### D1. Yazılı çıktılar

Yapman gerekenler, sırasıyla:

**1. Detaylı laboratuvar notları (her hafta):**
Hangi denemeyi neden yaptın, ne sonuç aldın, ne öğrendin. Bu hem senin için hem de sonradan paper yazarken kritik.

**2. Teknik blog yazıları (her 2-3 ayda bir):**
İlerlemeni topluluğa duyur. LessWrong, kendi blog, Medium. Geri bildirim alırsın, network kurarsın.

**3. arXiv preprint:**
Faz 1 sonuçlarını yazıp arXiv'e koy. Resmi yayın olmadan da insanlar görür ve okur. Format için ICML/NeurIPS template kullan.

**4. Konferans gönderimi:**
Faz 2 tamamlanınca ICLR, NeurIPS, ICML gibi yerlere gönder. Reddedilirse normaldir, geri bildirimle iyileştir.

### D2. Topluluk

Yalnız yapma. Bu projeyi ilerletmenin yolu:

- ML Twitter/X'te aktif ol. Diffusion, geometric DL araştırmacılarını takip et, ara sıra etkileşime gir.
- EleutherAI Discord'una katıl. Açık araştırma topluluğu, geri bildirim için harika.
- Yerel ML meetup'lara git.
- Bir-iki kişiyle düzenli görüş — danışman gibi olmasa da, fikir alışverişi için.

### D3. Fonlama / kariyer

12 ay sonra, kavram kanıtın varsa şu seçenekler açılır:

- **PhD başvurusu:** İyi bir preprint + somut prototip, başvuruda inanılmaz güçlü.
- **Endüstri araştırma laboratuvarı:** Anthropic, DeepMind, Meta FAIR, vs. araştırmacı pozisyonları. Ama önce yayın gerekir.
- **Bağımsız bağışlar:** Open Philanthropy, Long-Term Future Fund, Manifund. Küçük ama bağımsız çalışmaya yeter.
- **Startup:** Daha riskli, ama mimari ticari potansiyel taşıyorsa mümkün.

---

## Bölüm E: Sık Yapılan Hatalar (Önle)

Bunları okuyup içine sindir, çünkü her birine düşme ihtimalin yüksek:

**Hata 1: Önce büyük ölçek deneme.**
"100M parametre ile başlayayım" deme. 1M ile başla, çalıştır, sonra büyüt. Büyük model debug edemezsin.

**Hata 2: Hipotezi çok erken değiştirme.**
İlk denemede transformer'ı geçemezsen, hemen "mimari yanlış" diye değiştirme. Önce **implementasyon hatası** olup olmadığını dikkatlice ara. Çoğu zaman kavramsal değil, pratik problem var.

**Hata 3: Baseline'ı atlama.**
"Transformer baseline kurmak zaman kaybı" denir, ama yanlış. Baseline olmadan LDC sonuçları anlamsız.

**Hata 4: Tek başına çalışma.**
Düzenli geri bildirim olmadan ay sonra fark edersin ki yanlış yöne gitmişsin. Birinin kodunu okumasına ya da metodolojini eleştirmesine izin ver.

**Hata 5: Mükemmellik tuzağı.**
"Hibrit manifoldu mükemmel implement etmeden bir sonraki adıma geçmem" deme. Çalışan kötü versiyon, çalışmayan mükemmel versiyondan daha iyidir. Iterate.

**Hata 6: Aşırı kapsamlı ilk deney.**
"Bu deneyde hem hiperbolik geometriyi hem adaptive computation'ı hem de yeni gürültüyü test edeceğim." Hayır. Tek değişken değiştir, sonucu gör, sonra bir sonrakini.

**Hata 7: Bilim yerine mühendislik.**
LDC bir araştırma projesi. "Ne yaptığını" değil, "neden çalıştığını" anlamaya odaklan. Çalışan bir model bile yanlış sebeple çalışıyorsa değersizdir.

---

## Bölüm F: Bir Sonraki Adım (Bu Hafta)

Düşünme bitti, eylem zamanı. Bu hafta yapacakların:

**Pazartesi-Çarşamba:**
1. GitHub'da yeni bir repo aç: `ldc-research`
2. README.md'ye projenin amacını yaz (sadece sen okuyacak olsan bile)
3. Conda environment kur (yukarıdaki komutlar)
4. Wandb hesabı oluştur ve test et

**Perşembe-Cuma:**
5. SCAN datasetini indir, veriye bak. 100 örneği elle oku. Görevin ne olduğunu sezgisel anla.
6. Karpathy nanoGPT repo'sunu klonla, çalıştır. Eğer karakterleri tahmin eden bir model çalıştıramıyorsan, LDC'ye hazır değilsin demektir; önce bunu öğren.

**Hafta sonu:**
7. DDPM paper'ını oku (Ho et al. 2020). En az iki kez. Matematiği anlamadığın yerleri not et.
8. Bu rehbere geri dön, B bölümünü yeniden oku. Hangi adımda olduğunu kendine sor.

Bir sonraki hafta: ilk transformer baseline'ı yaz, SCAN'da eğit. Bu kadar.

---

## Son Söz

Bu proje muhtemelen başarısız olacak — istatistiksel olarak. Çoğu radikal mimari fikir başarısız olur. **Bu normaldir ve değerlidir.**

Ama başarısız olsa bile, sen şunu kazanırsın: derin teknik bilgi, somut bir araştırma çıktısı, ve "büyük problemler üzerinde gerçekten çalışma" deneyimi. Bunlar hayatta her şeyin önünü açar.

Başarılı olursa: insanlık adına gerçek bir katkı yapmış olursun.

İki sonuç da değerli. Sadece başla.

---

*Sorular geldikçe sor. Bu rehberi başlangıç noktası olarak gör — yola çıktıkça bu plan değişecek, değişmeli.*
