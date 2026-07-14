import torch 
from torch.utils.data import Dataset
from src.data.tokenizer import AuraTokenizer
from torch.utils.data import DataLoader



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
    


def create_dataloader(text: str, tokenizer: AuraTokenizer, max_length: int, stride: int, batch_size: int, shuffle: bool = True):
    dataset = AuraDataset(text, tokenizer, max_length, stride)
    
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True 
    )
    
    return dataloader


if __name__ == "__main__":
    tokenizer = AuraTokenizer()
    simple_text = "Aura projesi adım adım büyüyor ve harika bir veri hattına sahip oluyor."
    
    # DataLoader oluşturuyoruz: max_length=4, stride=1, batch_size=2
    dataloader = create_dataloader(
        text=simple_text,
        tokenizer=tokenizer,
        max_length=4,
        stride=1,
        batch_size=2,
        shuffle=False  # Test ederken sırayı karıştırmayalım ki çıktıyı rahat okuyalım
    )
    
    print(f"Toplam Batch (Grup) Sayısı: {len(dataloader)}")
    print("-" * 50)
    
    # İlk grubu (batch) çekip inceleyelim
    for x_batch, y_batch in dataloader:
        print("Batch Girdi Matrisi (X Boyutu):", x_batch.shape)
        print("Batch Hedef Matrisi (Y Boyutu):", y_batch.shape)
        print("\nGirdi Matrisi (X):")
        print(x_batch)
        print("\nHedef Matrisi (Y):")
        print(y_batch)
        
        # İlk batch'ten sonra döngüyü kıralım, tek bir tanesini incelemek yeterli
        break