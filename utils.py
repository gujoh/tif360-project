import os 
from PIL import Image 
from torch.utils.data import Dataset as TorchDataset
from matplotlib import pyplot as plt
from torchvision.transforms import functional as v_F
import torchvision.utils as vutils
import torch
import numpy as np

# Dataset to store the images. 
class Dataset(TorchDataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.total_imgs)
    
    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

def get_data(transform=None, batch_size=32, shuffle=True):
    paintings = Dataset("./data/monet_jpg", transform)
    photos = Dataset("./data/photo_jpg", transform)
    painting_loader = torch.utils.data.DataLoader(paintings, batch_size=batch_size, shuffle=shuffle)
    photo_loader = torch.utils.data.DataLoader(paintings, batch_size=batch_size, shuffle=shuffle)
    return paintings, photos, painting_loader, photo_loader
    
# Shows an image. 
def show_img(img):
    plt.imshow(np.transpose(vutils.make_grid(img, normalize=True), (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Shows all the images in a batch.
def show_batch(batch):
    plt.figure(figsize=(10,8))
    plt.imshow(np.transpose(vutils.make_grid(batch, padding=2, normalize=True), (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.show()