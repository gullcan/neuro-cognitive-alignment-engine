# Teknoloji ve Mülakat Rehberi

Bu dokümanın amacı kod ezberletmek değil, projeyi mimari kararlarıyla anlatabilecek seviyeye
getirmektir. Bir mülakatta her teknoloji için şu dört soruya cevap verebilmelisin:

1. Bu araç nedir?
2. Projede hangi sorunu çözüyor?
3. Neden bu aracı seçtim?
4. Sınırı veya alternatifi nedir?

## 1. Projeyi tek cümlede anlat

> Notion'daki dinamik günlük planı LangGraph ile stateful ajan akışlarına dönüştüren,
> Telegram üzerinden plan onayı ve görev takibi yapan, PostgreSQL/pgvector geçmişinden
> bağlam getirerek LLM tabanlı fakat güvenlik sınırları olan davranış geri bildirimi üreten
> event-driven bir FastAPI sistemi geliştirdim.

Bu cümlede projenin veri kaynağı, orkestrasyonu, kullanıcı kanalı, hafızası, LLM kullanımı
ve backend'i birlikte görünür.

## 2. Araçlar ve teknolojiler

### Python 3.12

**Nedir?** Uygulamanın programlama dili ve çalışma zamanı.

**Projede görevi:** FastAPI endpoint'leri, LangGraph node'ları, entegrasyon istemcileri,
repository katmanı ve testler Python ile yazıldı. Async/await sayesinde Notion, Telegram,
LLM ve veritabanı I/O işlemleri beklerken thread bloklanmıyor.

**Neden seçildi?** AI/LLM ekosistemi, Pydantic, LangGraph ve veri araçları Python'da güçlü.
Type hint desteği agent state sözleşmelerini görünür kılıyor.

**Mülakat cümlesi:** “I/O ağırlıklı entegrasyonları `async/await` ile yönettim; CPU-bound
bir iş olsaydı async'in tek başına çözüm olmayacağını, process veya ayrı worker
gerekeceğini biliyorum.”

### uv

**Nedir?** Python paket ve sanal ortam yöneticisi.

**Projede görevi:** `pyproject.toml` bağımlılıklarını çözer, `uv.lock` ile kesin sürümleri
kilitler, test ve uygulama komutlarını aynı ortamda çalıştırır.

**Neden seçildi?** Hızlıdır ve “benim makinemde çalışıyor” farkını azaltan reproducible
build sağlar.

**Bilmen gereken:** `pyproject.toml` kabul edilen sürüm aralıklarını, `uv.lock` ise çözülmüş
tam bağımlılık ağacını taşır.

### Pydantic

**Nedir?** Python type hint'lerinden runtime veri doğrulaması ve JSON schema üreten
kütüphane.

**Projede görevi:** `NormalizedInboundEvent`, `DailyPlan`, `ConversationDecision`,
`NeuroFeedback` ve API request/response modellerini doğrular. `extra="forbid"` beklenmeyen
alanları reddeder.

**Neden önemli?** LLM çıktısı düz metin yerine schema'ya bağlı veri olur. Bu, LLM'yi
deterministik yapmaz fakat çıktısının biçimini ve sınırlarını kontrol eder.

**Mülakat cümlesi:** “LLM output'u doğrudan iş kuralı olarak kullanmadım; önce Pydantic
structured output, sonra task membership ve confidence gibi deterministik kontrollerden
geçirdim.”

### FastAPI

**Nedir?** ASGI tabanlı modern Python web framework'ü.

**Projede görevi:**

- Telegram webhook'unu alır.
- Daily-plan ve task-monitor scheduler endpoint'lerini sunar.
- Outbox delivery endpoint'ini korur.
- Liveness ve readiness endpoint'lerini sağlar.
- OpenAPI/Swagger sözleşmesini otomatik üretir.

**Neden seçildi?** Async desteği, Pydantic entegrasyonu, dependency injection ve otomatik
OpenAPI sağlar.

**Liveness vs readiness:**

- Liveness: proses yaşıyor mu?
- Readiness: graph başladı mı ve veritabanına erişiliyor mu?

### Uvicorn ve ASGI

**Nedir?** Uvicorn, FastAPI uygulamasını çalıştıran ASGI sunucusudur.

**Projede görevi:** Render container'ında HTTP trafiğini kabul eder. ASGI, async request
işlemeyi mümkün kılar.

**WSGI farkı:** WSGI geleneksel senkron Python web arayüzüdür; ASGI async, WebSocket ve uzun
süren bağlantılara daha uygundur.

### LangGraph

**Nedir?** LLM/agent iş akışlarını node, edge, state ve checkpoint kavramlarıyla yöneten
stateful orkestrasyon framework'ü.

**Projede görevi:** Her event önce claim node'una gelir; event türüne göre daily plan,
monitor, plan decision, behavior veya conversation dalına yönlenir. Node çıktıları ortak
`WorkflowState` üzerinde birleşir.

**Neden basit chain değil?** Sistem koşullu dallara, tekrar üretim döngüsüne, farklı son
durumlara, checkpoint'e ve idempotent event işlemeye ihtiyaç duyuyor.

**State Management:** Graph node'ları birbirine gizli global değişkenlerle değil, serialize
edilebilir typed state ile veri aktarır. Her invocation başında volatile alanlar resetlenir.

**Checkpoint:** Graph'ın çalışma durumunu saklar. Bir checkpoint kullanıcı davranış
geçmişinin kaynağı değildir; execution recovery içindir.

### Multi-Agent tasarımı

Bu projede agent, ayrı bir prompt ve sorumluluğu olan uzman graph node'udur:

- Planner Agent plan üretir.
- Monitor Agent zamanı ve action kayıtlarını değerlendirir.
- Conversation Agent serbest mesajı yorumlar.
- Neuro-Behavioral Agent evidence-bound feedback üretir.
- Safety Critic model çıktısını denetler.

**Önemli savunma:** “Her şeyi otonom ajanlara bırakmadım. Kontrol akışı deterministik,
ajanların giriş/çıkışları schema-bound. Bu yüzden sistemi test edebiliyorum.”

### Notion API

**Nedir?** Notion workspace verilerini programatik okumaya/yazmaya yarayan HTTP API.

**Projede görevi:** Günün `Window` tarihine uyan ve `Archived` olmayan görevleri okur.
Pagination uygular ve property şemasını doğrular.

**Dinamik veri ne demek?** Görevler uygulama koduna gömülü değildir. Kullanıcı Notion'da
planı değiştirdiğinde bir sonraki 15 dakikalık sync bunu görür.

**Schema contract neden gerekli?** “Task” yanlışlıkla silinirse sessizce boş veri üretmek
yerine `NotionSchemaError` oluşur. Bu fail-fast davranıştır.

### Telegram Bot API ve webhook

**Nedir?** Telegram botlarıyla mesaj ve callback alışverişi sağlayan API.

**Webhook yaklaşımı:** Telegram update'i public HTTPS endpoint'e POST eder. Polling gibi
uygulamanın sürekli Telegram'a soru sorması gerekmez.

**Projede güvenlik:**

- secret-token header doğrulanır;
- yalnızca ayarlanmış chat ID kabul edilir;
- callback payload desteklenen action enum'una çevrilir;
- Telegram update ID idempotency anahtarı olur.

**Inline buttons:** Başladım, Bitirdim, Takıldım gibi yapılandırılmış self-report üretir.
Serbest mesajlar Conversation Agent tarafından ayrıca yorumlanır.

### Groq / OpenAI Responses API

**Nedir?** Büyük dil modellerinden metin veya structured output almak için kullanılan API.

**Projede görevi:** Plan, davranış feedback'i ve serbest konuşma kararı üretir.

**Neden structured output?** Modelden “bir JSON yaz” demek yalnızca prompt isteğidir.
Structured output ise `ConversationDecision` gibi bir JSON schema'yı API seviyesinde
uygular ve Pydantic tekrar doğrular.

**Fallback:** Groq key varsa önce Groq, yoksa OpenAI, o da yoksa rule-based provider
seçilir. Timeout, quota veya geçersiz çıktı durumunda dış etkileşim kaybolmaz.

**LLM'nin yapmadığı işler:** Scheduler zamanı, authentication, SQL transaction, task
membership kontrolü ve Telegram retry mekanizması LLM'ye ait değildir.

### Prompt engineering

Prompt yalnızca ton tarif etmez. Şunları içerir:

- görevin sınırı;
- kullanılabilecek evidence;
- action üretme koşulları;
- belirsizlik davranışı;
- yasak klinik/biyolojik iddialar;
- kullanıcı mesajı ve task alanlarının “untrusted data” olduğu kuralı;
- çıktı schema'sı.

**Prompt injection yaklaşımı:** Notion task başlığı veya Telegram mesajı developer
instruction olarak değil veri olarak gönderilir. Model çıktısı yine deterministik
kontrollerden geçer.

### PostgreSQL

**Nedir?** İlişkisel, transaction destekli açık kaynak veritabanı.

**Projede görevi:** Inbox, domain event, plan, outbox, behavior memory ve LangGraph
checkpoint verilerini kalıcı tutar.

**Neden yalnızca vector database değil?** Sistem transaction, unique constraint, tarih
sorgusu, lease ve status transition gibi ilişkisel garantilere ihtiyaç duyuyor. pgvector
sayesinde aynı veritabanı vector retrieval da yapıyor.

### SQLAlchemy Async

**Nedir?** Python ORM ve SQL toolkit'i.

**Projede görevi:** Typed model mapping, async session ve transaction yönetimi sağlar.
Repository sınıfları graph node'larını SQL ayrıntısından ayırır.

**Transaction sınırı:** `async with session.begin()` içindeki değişikliklerin tamamı commit
olur veya hata halinde rollback edilir.

### JSONB

**Nedir?** PostgreSQL'in binary JSON türü.

**Projede görevi:** Event payload, plan ve inline button gibi evrilebilir dokümanları tutar.

**Neden her şey kolon değil?** Transport metadata ve structured plan zamanla genişleyebilir.
Kimlik, tarih, status ve idempotency alanları ise sorgu/constraint gerektiği için normal
kolonlardır.

**Trade-off:** JSONB esnektir ama kontrolsüz kullanılırsa ilişkisel şema avantajını
kaybettirir.

### pgvector

**Nedir?** PostgreSQL'e vector veri tipi ve benzerlik operatörleri ekleyen extension.

**Projede görevi:** 32 boyutlu normalize edilmiş davranış bağlam vektörlerini saklar ve
cosine distance ile benzer geçmiş episode'ları getirir.

**Vektördeki bilgiler:** action türü, priority, commitment tier, weekday, saat dilimi,
cognitive load, estimated duration ve evidence requirement.

**Dürüst ayrım:** Bu bir metin embedding'i değildir; engineered feature vector'dür.
“Kullanıcının beynini ölçüyor” veya “semantic olarak her şeyi anlıyor” denmez.

### HNSW index

**Nedir?** Yaklaşık en yakın komşu araması için graph tabanlı vector index.

**Projede görevi:** PostgreSQL'de cosine similarity aramasını büyüyen memory tablosunda
hızlandırır.

**Trade-off:** Daha hızlı sorgu karşılığında ek disk/memory ve yaklaşık sonuç kullanır.
Küçük test verisinde SQLite exact cosine hesabı kullanılıyor.

### RAG

**Nedir?** Retrieval-Augmented Generation; model cevap üretmeden önce dış kaynaktan ilgili
bağlam getirme deseni.

**Bu projede nasıl?**

1. Güncel task action kaydedilir.
2. Task-specific event count'ları çıkarılır.
3. Benzer behavioral episode'lar pgvector'dan getirilir.
4. Bu evidence LLM prompt'una eklenir.
5. Feedback evidence sınırında üretilir.

**Dürüst tanım:** Bu document RAG değildir. Chunking, PDF indexing ve semantic text
embedding yoktur. Davranış episode retrieval'ı vardır.

### Alembic

**Nedir?** SQLAlchemy için veritabanı migration aracı.

**Projede görevi:** Operational schema'yı revision zinciri halinde upgrade/downgrade eder.
JSON → JSONB geçişi ve pgvector/HNSW kurulumu migration ile yönetilir.

**Neden create_all değil?** Production schema değişiklikleri versiyonlu, review edilebilir
ve tekrar uygulanabilir olmalıdır. `create_all` mevcut kolonları güvenli şekilde evrimleştirmez.

### Inbox/idempotency pattern

**Problem:** Telegram veya scheduler aynı event'i retry edebilir.

**Çözüm:** `inbound_events` üzerinde source + source_event_id unique constraint'i vardır.
Claim edilen event tamamlanınca aynı event graph'ın devamına geçmez.

**Idempotent ne demek?** Aynı isteğin birden fazla uygulanması, sistemin mantıksal sonucunu
bir kez uygulanmış gibi bırakır.

### Durable outbox

**Problem:** Veritabanına “task completed” yazıp Telegram gönderirken proses çökerse iki
sistem tutarsız kalabilir.

**Çözüm:** Gönderilecek mesaj önce database içinde outbox kaydı olur. Ayrı dispatcher lease
alarak Telegram'a yollar ve sonucu işaretler.

**Garanti:** Uygulama içinde logical duplicate önlenir; Telegram sınırında at-least-once
delivery vardır. Exactly-once iddiası yapılmaz. Domain write ile outbox write ayrı
repository transaction'larıdır; bu nedenle tam atomic “transactional outbox” garantisi
iddia etmiyorum. Production sürümünde ikisini tek transaction'a alırdım.

### Docker ve multi-stage build

**Nedir?** Uygulamayı kod, runtime ve bağımlılıklarla taşınabilir image haline getirir.

**Projede görevi:** Builder stage uv ile locked environment oluşturur; runtime stage yalnız
gerekli virtual environment ve migration dosyalarını alır. Uygulama root olmayan
`appuser` ile çalışır.

**Neden multi-stage?** Build araçlarını final image'a taşımadan daha küçük ve daha az
saldırı yüzeyli image üretir.

### Render

**Nedir?** Container tabanlı uygulama hosting platformu.

**Projede görevi:** FastAPI'yi public HTTPS URL'de çalıştırır; Telegram webhook bu URL'ye
gelir. `render.yaml` deployment konfigürasyonudur.

**Free-tier sınırı:** Cold start ve SLA yoktur. Bu nedenle zamanlama “yaklaşık”tır ve
HTTP retry/idempotency kullanılır.

### Neon

**Nedir?** Serverless managed PostgreSQL platformu.

**Projede görevi:** Render'ın ephemeral diskinden bağımsız kalıcı operational data,
pgvector memory ve checkpoint saklar.

**Neden ayrı database?** Container yeniden deploy olduğunda yerel disk kaybolabilir.

### GitHub Actions

**Projede üç kullanım:**

1. 07:35 günlük Notion planı.
2. Her 15 dakikada Notion refresh + task monitor.
3. Push/PR üzerinde Ruff, mypy ve pytest CI.

Scheduler endpoint'leri internal API key ile korunur. Workflow concurrency aynı işin üst
üste kontrolsüz koşmasını önler.

### Ruff

**Nedir?** Hızlı Python formatter ve linter.

**Projede görevi:** Stil tutarlılığı, import düzeni ve statik kod hatalarını kontrol eder.

### mypy strict

**Nedir?** Python static type checker.

**Projede görevi:** Production `src` kodunda eksik/uyumsuz tipleri CI öncesi yakalar.
Agent state ve provider protocol'leri için özellikle değerlidir.

### pytest

**Nedir?** Python test framework'ü.

**Projede görevi:** Workflow route'ları, idempotency, migrations, Notion parsing, Telegram
security, outbox retry, LLM fallback ve conversation intent davranışlarını doğrular.

**Test stratejisi:** Dış API'ler stub/fake ile izole edilir; SQLite hızlı integration test
ortamı sağlar; migration testleri revision round-trip kontrol eder.

### structlog

**Nedir?** Yapılandırılmış log kütüphanesi.

**Projede görevi:** Event adı ve alanları ayrıştırılabilir biçimde yazar. Error loglarında
secret veya ham provider URL'si taşımamaya dikkat edilir.

### HTTPX

**Nedir?** Async destekli HTTP client.

**Projede görevi:** Notion ve Telegram çağrılarını timeout ve kontrollü error handling ile
yapar.

## 3. Üç temel iş akışını anlat

### Sabah planı

1. GitHub Actions authenticated daily-plan endpoint'ini çağırır.
2. FastAPI event'i normalize eder.
3. Inbox event ID'yi claim eder.
4. Planner branch Notion görevlerini çeker.
5. Planner Agent typed `DailyPlan` üretir.
6. Material content hash eski planla karşılaştırılır.
7. Değişmiş plan pending olarak kaydedilir.
8. Outbox Telegram onay mesajını taşır.
9. Kullanıcı onaylayınca plan active/approved olur.

### Zamanı gelen görev

1. 15 dakikalık workflow önce Notion'u refresh eder.
2. Değişiklik yoksa approval korunur.
3. Monitor approved plan ve bugünkü action event'lerini okur.
4. Saat/grace/expected duration kurallarını değerlendirir.
5. Stable outbox key ile yalnız gerekli reminder oluşur.
6. Buton cevabı yeni behavior event ve memory episode yaratır.
7. Evidence retrieval sonrası feedback üretilir.

### Serbest Telegram mesajı

1. Webhook secret ve chat ID doğrulanır.
2. Mesaj check-in event olarak kaydedilir.
3. Approved plan, activity ve focus task bulunur.
4. Conversation Agent `ConversationDecision` üretir.
5. Explicit action + exact task + yeterli confidence varsa event kaydedilir.
6. “Yapmak istemiyorum” reluctance olarak kalır; skip olmaz.
7. Kısa reply outbox üzerinden gönderilir.

## 4. Mülakat soruları ve güçlü cevaplar

### “Bu neden multi-agent?”

Çünkü planlama, zaman kontrolü, serbest mesaj yorumlama, behavioral feedback ve safety
review farklı input/output sözleşmelerine sahip uzman sorumluluklar. Hepsini tek prompt'a
koymak yerine ayrı LangGraph node'larına böldüm. Kontrol akışı deterministik kaldı.

### “Neden LangGraph, düz Python fonksiyonları değil?”

Düz fonksiyonlarla da ilk sürüm yapılabilirdi. LangGraph'ı dallanan event türleri, typed
shared state, checkpoint, feedback retry loop ve thread isolation ihtiyacı için seçtim.
Framework'ü sırf agent kelimesi için kullanmadım.

### “State ile memory arasındaki fark?”

State, bir graph invocation sırasında node'lar arasında taşınan çalışma verisidir.
Checkpoint, bu execution state'in kalıcı snapshot'ıdır. Behavioral memory ise geçmiş
episode retrieval'ı için domain verisidir. Birbirlerinin yerine geçmezler.

### “LLM hallucination riskini nasıl yönettin?”

Structured output, evidence references, task membership, confidence threshold, forbidden
claim kontrolleri, Safety Critic ve rule-based fallback kullandım. LLM'yi source of truth
yapmadım.

### “LLM tamamen kapanırsa ne olur?”

Groq/OpenAI exception veya invalid schema üretirse `ResilientIntelligenceProvider` local
rule-based provider'a döner. Reminder, button ve persistence zaten deterministik olduğu için
çekirdek sistem çalışmaya devam eder.

### “RAG nerede?”

Task action sonrası event count'ları ve pgvector'daki benzer behavior episode'lar
retrieval ile alınarak feedback prompt'una ekleniyor. Bu behavioral-context RAG'dir;
document RAG veya semantic text embedding değildir.

### “Neden pgvector?”

Relational event ve transaction verisi zaten PostgreSQL'de. Aynı database içinde vector
query yapmak operasyonel karmaşıklığı azaltıyor. Ayrı vector DB bu ölçek için gereksizdi.

### “32 boyut nereden geliyor?”

Bu learned embedding değil. Action, priority, commitment tier, weekday, time bucket,
cognitive load ve süre gibi observable features için tasarlanmış sabit bir vektör.
Normalize edilip cosine similarity ile aranıyor.

### “Exactly-once sağlıyor musun?”

Uygulama içinde unique keys ile logical idempotency sağlıyorum. Telegram provider sınırında
exactly-once garanti etmiyorum; outbox at-least-once çalışıyor. Accept ile sent marker
arasındaki crash duplicate fiziksel mesaj üretebilir.

### “Plan polling neden spam üretmiyor?”

Approval token material plan content'inden türetiliyor. Aynı content geldiğinde status
preserve ediliyor ve outbox mesajı üretilmiyor. İçerik değişirse yeni token ve yeniden human
approval gerekiyor.

### “Human-in-the-loop neden var?”

Notion değişikliği otomatik olarak davranış sözü sayılmamalı. Kullanıcı Telegram'da planı
onaylamadan monitor onu active kabul etmiyor. Bu hem UX hem güvenlik sınırı.

### “Webhook'u nasıl güvene aldın?”

Telegram secret-token header, configured chat ID allowlist ve callback enum parsing
kullandım. Internal scheduler endpoint'leri ayrı API key ister. Secret'lar Git'te yoktur.

### “Database migration yaklaşımın?”

Alembic revision chain kullanıyorum. PostgreSQL production'da `create_all` yasak; startup
`alembic upgrade head` çalıştırıyor. JSONB ve vector değişiklikleri upgrade/downgrade
testleriyle doğrulandı.

### “Neden event-driven?”

Sistemin doğal girişleri Telegram update, scheduler tick ve Notion plan request event'leri.
Her event normalize edilip idempotent işleniyor. Bu, transport ile domain logic'i ayırıyor.

### “En zor bug neydi?”

Monitor yalnız database'deki sabah planına bakıyordu. Sabah plan boşsa gün içinde Notion'a
eklenen görevler görünmüyordu. Her monitor cycle öncesine Notion refresh ekledim; fakat
unchanged refresh'in approved planı pending'e çevirmemesi için content-aware persistence
tasarladım.

### “Bir başka zor deployment problemi?”

Render Docker command quoting ve Alembic config path problemi yaşandı. Shell string yerine
portable Python entrypoint kullandım; entrypoint migration config'ini package/install
konumundan bağımsız çözüp upgrade sonrası Uvicorn'u başlatıyor.

### “Observability yeterli mi?”

Şu an structured logs ve liveness/readiness var. Production next step OpenTelemetry trace,
metrics, alerting, correlation ID ve outbox lag dashboard'udur.

### “Bu neuroscience projesi mi?”

Klinik ölçüm yapan neuroscience sistemi değil. Öğrenme, cue-action ve habit gibi genel
kavramları dikkatli dil sınırında kullanıyor. Dopamin veya korteks değişimi ölçtüğünü iddia
etmiyor. Esas mühendislik değeri stateful orchestration ve evidence-bounded feedback.

### “Ölçek nasıl büyütülür?”

API ve worker'ı ayırırım, managed scheduler/queue eklerim, tenant-scoped auth ve row-level
security tasarlarım, connection pool ve rate limit uygularım, outbox consumer sayısını
artırırım, tracing/metrics eklerim.

### “Neden free-tier mimari?”

Portföy amacıyla maliyetsiz çalışan bir vertical slice hedefledim. Cold-start ve SLA
sınırlarını gizlemedim; retry ve idempotency ile etkisini azalttım. Production için paid
always-on compute ve durable scheduler gerekir.

## 5. Whiteboard'da çizmen gereken şema

Beş kutu çiz:

```text
Notion ----\
            -> FastAPI -> LangGraph -> PostgreSQL/pgvector
Telegram --/                |                |
                            LLM          Outbox -> Telegram
```

Sonra şu sırayla anlat:

1. Dış sistemleri FastAPI normalize ediyor.
2. LangGraph event'i route ediyor.
3. PostgreSQL source of truth.
4. LLM yalnız bounded generation.
5. Outbox güvenilir teslimat.
6. Checkpoint ve behavioral memory ayrı.

## 6. 60 saniyelik proje sunumu

> Bu projede Notion günlük planını dinamik olarak okuyan ve Telegram üzerinden kullanıcıyla
> çalışan stateful bir multi-agent sistem geliştirdim. FastAPI webhook ve scheduler
> sınırlarını yönetiyor; LangGraph Planner, Monitor, Conversation, Neuro-Behavioral ve
> Safety Critic node'larını typed state üzerinden orkestre ediyor. Kullanıcı action'ları
> PostgreSQL'de event olarak saklanıyor, 32 boyutlu observable context vector'leri pgvector
> ile benzer geçmiş episode'ları getiriyor. LLM yalnız structured plan ve feedback üretiyor;
> task state, authentication ve delivery deterministik kodda. Duplicate event'leri inbox
> pattern, mesaj güvenilirliğini durable leased outbox ile yönettim. Sistem Render, Neon ve
> GitHub Actions üzerinde ücretsiz çalışan bir deployment'a sahip ve Ruff, strict mypy,
> pytest CI ile doğrulanıyor.

## 7. Üç dakikalık sunum iskeleti

1. **Problem:** Plan yazmak ile uygulamak arasındaki kopukluk.
2. **Kullanıcı akışı:** Notion planı → Telegram onayı → saatli takip → feedback.
3. **Mimari:** FastAPI + LangGraph + PostgreSQL/pgvector + LLM + outbox.
4. **Agent ayrımı:** Planner, Monitor, Conversation, Behavioral, Critic.
5. **Enterprise özellikler:** typed contracts, idempotency, migrations, checkpoint,
   fallback, security headers, CI.
6. **Dürüst sınırlar:** single-user, self-report, free-tier, no clinical measurement.
7. **Next step:** multi-tenancy, worker separation, evals, observability.

## 8. Ezberlememen, gerçekten anlaman gereken noktalar

- LLM ile agent aynı şey değildir; agent goal, state, tools ve control loop bağlamında
  tanımlanır.
- “Stateful” demek yalnız chat history tutmak değildir; workflow durumunun açık ve kalıcı
  yönetilmesidir.
- Vector memory gerçeklerin kaynağı değil retrieval index'idir.
- Confidence bir gerçeklik garantisi değildir; yalnız mutation gate'in bir parçasıdır.
- Async daha hızlı CPU hesabı yapmaz; I/O bekleme verimliliği sağlar.
- Idempotency ile exactly-once aynı şey değildir.
- Checkpoint ile domain event aynı şey değildir.
- Prompt güvenlik katmanlarından yalnız biridir.
- Free tier deployment production SLA anlamına gelmez.
- En güçlü mülakat cevabı, yaptığın trade-off'u ve sınırı dürüstçe açıklayabilmektir.

## 9. Kendini test et

Aşağıdaki soruları dokümana bakmadan cevapla:

1. Aynı Telegram update'i iki kez gelirse hangi tablo engeller?
2. Aynı Notion planı 15 dakikada bir okununca approval neden sıfırlanmaz?
3. Conversation Agent “yapmak istemiyorum” mesajını neden skipped yapmaz?
4. Task ID'nin LLM tarafından uydurulmasını hangi kontrol engeller?
5. LangGraph checkpoint ile `domain_events` arasındaki fark nedir?
6. HNSW neyi hızlandırır ve karşılığında ne ödersin?
7. Telegram delivery neden tam exactly-once değildir?
8. LLM quota dolarsa kullanıcı neden tamamen cevapsız kalmaz?
9. Alembic neden `create_all` yerine kullanıldı?
10. Bu sistem neden klinik neuroscience ürünü değildir?

Bu on soruya net cevap verebiliyorsan projenin yalnız arayüzüne değil, mimarisine de
hakimsin.
