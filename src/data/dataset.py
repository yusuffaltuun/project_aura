import torch 
from torch.utils.data import Dataset
from src.data.tokenizer import AuraTokenizer


class AuraDataset(Dataset):
    def __init__(self,text , tokenizer:AuraTokenizer , max_length:int,stride:int):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
    def __len__(self):
      return len(self.input_ids)

    def __getitem__(self,idx):
              return self.input_ids[idx] , self.target_ids[idx]



if __name__ == "__main__":
    tokenizer = AuraTokenizer()
    simple_text = "Aura projesi adım adım büyüyor."
    
    dataset = AuraDataset(simple_text, tokenizer, max_length=3, stride=2)
    
    print(f"Toplam Veri Çifti Sayısı: {len(dataset)}")
    print("-" * 40)
    
    # İlk iki çifti döngüyle çekip inceleyelim
    for i in range(min(2, len(dataset))):
        x, y = dataset[i]
        print(f"Örnek {i + 1}:")
        print(f"  Girdi (x) Tensor: {x}")
        print(f"  Hedef (y) Tensor: {y}")
        print(f"  Girdi Kelimeleri: {tokenizer.decode(x.tolist())}")
        print(f"  Hedef Kelimeleri: {tokenizer.decode(y.tolist())}")
        print("-" * 40)