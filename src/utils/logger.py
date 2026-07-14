import logging
import os
from datetime import datetime

# Logların kaydedileceği klasörü güvenli bir şekilde oluşturuyoruz
os.makedirs("logs", exist_ok=True)

# Günlük log dosyası adı (Örn: logs/aura_2026-07-14.log)
log_file = os.path.join("logs", f"aura_{datetime.now().strftime('%Y-%m-%d')}.log")

# Profesyonel log formatı:
# [Zaman Damgası] - [Seviye] - [Hangi Dosya:Hangi Satır] - Mesaj
log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

# Ana logging konfigürasyonunu yapıyoruz
logging.basicConfig(
    level=logging.INFO, # Bilgilendirme ve üzerindeki tüm seviyeleri (WARNING, ERROR) logla
    format=log_format,
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"), # Logları dosyaya yaz
        logging.StreamHandler()  # Aynı zamanda terminal ekranına da bas
    ]
)

# Proje genelinde import edip kullanacağımız logger nesnesini oluşturuyoruz
logger = logging.getLogger("AURA")