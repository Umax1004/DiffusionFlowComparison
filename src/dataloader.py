import torch
from datasets import load_dataset
from torchvision import transforms 

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self,
                 split: str='train',
                 transform=transforms.Compose([
                    transforms.Resize((128, 128)),  # Resize to 128x128
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
                ])):
        self.dataset = load_dataset("korexyz/celeba-hq-256x256", cache_dir="data/celeba-hq", split=split)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['image']
        if self.transform:
            img = self.transform(img)
        return img