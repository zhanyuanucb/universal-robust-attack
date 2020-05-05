import torch
from torch.utils.data import Dataset

def get_clipped_loss(loss_fn, min_val, max_val):
    def clipped_loss(logits, labels):
        loss = loss_fn(logits, labels)
        return torch.clamp(loss, min_val, max_val)
    return clipped_loss

class TargetClassSet(Dataset):
    
    def __init__(self, data_dir, transform=None):
        self.data, self.targets = torch.load(data_dir)
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return self.transform(img), target
    
    def __len__(self):
        return self.data.size(0)