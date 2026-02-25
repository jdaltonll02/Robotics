import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SimulationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.label_dir, self.labels[idx]))
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        label = torch.from_numpy(np.array(label)).long()
        return img, label
