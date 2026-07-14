import torch
import tiktoken
from src.config.settings import settings
from src.utils.logger import logger
import re





def _clean_text(self, txt: str) -> str:
        if not txt:
            return ""
        # 1. Görünmez kontrol karakterlerini temizleyelim
        txt = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', txt)
        # 2. Metnin başındaki ve sonundaki gereksiz boşlukları kırpalım
        return txt.strip()

class AuraTokenizer:
    def __init__(self) -> None:
        self.encoder = tiktoken.get_encoding("gpt2")
        logger.info("AuraTokenizer (gpt) başarıyla yüklendi")
        
    def encode(self,txt):
        if not txt:
            return []
        
        txt = self._clean_text(txt)
        tokens = self.encoder.encode(txt)
        if len(tokens) > settings.model.context_length:
            tokens = tokens[:settings.model.context_length]
            logger.warning("Metin baglam sinirini astigi icin kirpildi!")

        return tokens
        
    def decode(self,tokens):
        if not tokens:
            return ""
        text = self.encoder.decode(tokens)
        return text
    
    def _clean_text(self, txt: str) -> str:
        if not txt:
            return ""
        # 1. Görünmez kontrol karakterlerini temizleyelim
        txt = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', txt)
        # 2. Metnin başındaki ve sonundaki gereksiz boşlukları kırpalım
        return txt.strip()
        













# TEST KODLARI
if __name__ == "__main__":
    print("Mevcut Ayarlar:", settings.__dict__.keys())
    # Sınıfımızı ayağa kaldıralım
    tokenizer = AuraTokenizer()
    
    # Başında ve sonunda boşluklar, ortasında ise görünmez \x00 karakteri olan kirli bir metin:
    test_metni = "   Merhaba Yusuf\x00, Aura projesi ilk adımlarını başarıyla atıyor!   "
    print(f"\n[Orijinal Metin]: {test_metni}")
    
    # Sayılara çevirelim (encode)
    sayilar = tokenizer.encode(test_metni)
    print(f"[Sayılara Dönüştü (Tokens)]: {sayilar}")
    
    # Tekrar metne çevirelim (decode)
    cozulmus_metin = tokenizer.decode(sayilar)
    print(f"[Geri Çözülen Metin]: {cozulmus_metin}\n")





