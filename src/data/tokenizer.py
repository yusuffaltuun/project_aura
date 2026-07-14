import torch
import tiktoken
from src.config.settings import settings
from src.utils.logger import logger
import re



class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}   
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
    ]   
        preprocessed = [item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
tokenizer = tiktoken.get_encoding("gpt2")



class AuraTokenizer:
    def __init__(self) -> None:
        self.encoder = tiktoken.get_encoding("gpt2")
        logger.info("AuraTokenizer (gpt) başarıyla yüklendi")
        
    def encode(self,txt):
        if not txt:
            return []
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
        



# TEST KODLARI
if __name__ == "__main__":
    print("Mevcut Ayarlar:", settings.__dict__.keys())
    # Sınıfımızı ayağa kaldıralım
    tokenizer = AuraTokenizer()
    
    test_metni = "Merhaba Yusuf, Aura projesi ilk adımlarını başarıyla atıyor!"
    print(f"\n[Orijinal Metin]: {test_metni}")
    
    # Sayılara çevirelim (encode)
    sayilar = tokenizer.encode(test_metni)
    print(f"[Sayılara Dönüştü (Tokens)]: {sayilar}")
    
    # Tekrar metne çevirelim (decode)
    cozulmus_metin = tokenizer.decode(sayilar)
    print(f"[Geri Çözülen Metin]: {cozulmus_metin}\n")





