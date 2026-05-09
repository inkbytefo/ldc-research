# Planned Features

İleride değerlendirilecek mimari fikirler. Bu doküman bir taahhüt değil, **kaybolmaması gereken araştırma fikirlerinin** tutulduğu yerdir. Her fikir öneren tarafından kısaca yazılır; gerçekten geliştirilmeden önce ayrı bir tasarım dokümanı haline gelir.

Format: her madde **(seviye, faz, fırsat)** ile birlikte.

---

## PF-1: Mimari-yerli araç kullanımı (Architecture-native tool use)

- **Seviye:** Yüksek potansiyel, özgün katkı
- **Önerilen faz:** v3 (Faz 7 tamamlandıktan sonra)
- **Fırsat:** AR modellerinin yapısal olarak yapamadığı bir şey

### Fikir

Standart LLM'ler araçları "metin akışında özel token emit et → tool çıktısını metne enjekte et" şeklinde kullanır (Toolformer, ReAct). Bu yüzeysel bir entegrasyon: tool çağrıldıktan sonra modelin önceki düşüncesini geri dönüp düzeltmesi zordur, çünkü token üretildi gitti.

LDC'de **concept graph'taki bir slot gerçekten hesaplama yapabilir.** Yani:

- Bir "calculator slot"un forward pass'i sembolik hesaplama (Python `eval`, SymPy, kod execution).
- Diffusion refinement sırasında model "şu slotta sayısal cevap olmalı" sinyali verir.
- Slot deterministik bir araç çağrısıyla doldurulur (latent vektörü değil, gerçek değeri tutar).
- **Kalan diffusion adımları bu sabitlenmiş slot üzerine düşünür** — bu mimarinin kendi iteratif yapısının kazandırdığı şey.

### Neden LDC için doğal

1. Concept graph zaten **tipli, slot-yapılı**. "Compute slot" ekstra bir tip olur.
2. Diffusion **iteratif**. Tool çıktısı geldikten sonra graph'ın geri kalanı yeniden adapte olur. AR'da bu yapısal olarak imkansız.
3. Adaptive computation (Faz 6) ile uyumlu: zor problemler daha çok tool çağırır, kolay problemler hiç çağırmaz.

### Olası araç tipleri

- **Calculator** — aritmetik, cebir (SymPy)
- **Python interpreter** — kod yürütme, veri analizi
- **Knowledge graph query** — Wikidata, ontoloji çağrıları
- **External search** — RAG benzeri ama concept-level (token-level değil)
- **Symbolic solver** — SAT, ILP, Prolog

### Açık problemler

- Discrete tool call ile continuous diffusion arasındaki köprü (Gumbel-softmax? Hard stopping?)
- Eğitim sinyali: tool çağrısı backprop edilemiyor — RL/REINFORCE veya distillation gerekir
- Hangi slot'un "compute slot" olduğunu kim belirler? Sabit mi, öğrenilebilir mi?
- Latency: deterministic tool call diffusion adımını bloklar

### Bağımlılıklar

- Faz 1-4 tamamlanmış olmalı (LDC çalışıyor mu, biliyor olmalıyız)
- Faz 5 (graph yapısı zenginleştirildi)
- Faz 6 (adaptive halting var — tool ne zaman çağrılacağını öğrenmek için)

### İlk deney önerisi

GSM8K küçük subset üzerinde:
- LDC v2 (no tools) — Ay 5 sonucu
- LDC v2 + AR-style calculator tool (Toolformer benzeri)
- LDC v2 + slot-level calculator (PF-1)

Eğer (3) > (2) > (1), bu ciddi bir bilimsel katkıdır.

---

## PF-2: Toroidal manifold bileşeni

- **Seviye:** Orijinal mimaride var, ertelendi
- **Önerilen faz:** v3
- **Fırsat:** Döngüsel kavramlar (zaman, müzik, mevsim) için doğal geometri

`docs/latent-diffusion-cognition-mimarisi.md` Bölüm 3.1'deki orijinal hibrit manifold tasarımının üçüncü bileşeni. v2'de kapsam dışı bırakıldı çünkü Öklid+hiperbolik kombinasyonunun değer üretip üretmediğini önce kanıtlamamız gerekiyor.

Eklemek için Faz 5 sonrası, hiperbolik bileşenin somut katkısı görülmüş olmalı.

---

## PF-3: Çift katmanlı diffusion (anlam + dil)

- **Seviye:** Orijinal mimaride var, ertelendi
- **Önerilen faz:** v4 (gerekirse)
- **Fırsat:** Token-level diffusion + concept-level diffusion

Orijinal tasarımda decoder'da da diffusion vardı. `ldc-dogrulama-raporu.md` Bölüm 4.1 kararı: SONAR-LLM bulgusu gereği AR decoder kullanıyoruz. Ama "düşük entropili" çıktı tipleri (kod, matematik) için decoder-side diffusion tekrar değerlendirilebilir.

CCDD (2026) bunun benzerini token uzayında yaptı. Bizim concept-level + token-level kombinasyonumuz farklı olur.

---

*Bu dosyaya yeni fikir ekleyen: tarih + kim + 3-5 satır gerekçe yazsın. Detaylı tasarım gerekirse `docs/proposals/` altında ayrı doküman aç.*
