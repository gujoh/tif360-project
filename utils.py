import os 
from PIL import Image 
from torch.utils.data import Dataset as TorchDataset
from matplotlib import pyplot as plt
from torchvision.transforms import functional as v_F
import torchvision.utils as vutils
import torch
import numpy as np
import glob
import torch.nn as nn

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

class CycleDataset(TorchDataset):
    def __init__(self, root, transform=None, unaligned=False):
        self.transform = transform 
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root + '/monet_jpg') + '/*.*')[:301])
        self.files_B = sorted(glob.glob(os.path.join(root + '/photo_jpg') + '/*.*')[:301])
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
        
    def __getitem__(self, idx):
        image_A = Image.open(self.files_A[idx % len(self.files_A)])
        if self.unaligned: 
            image_B = Image.open(self.files_B[np.random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[idx % len(self.files_B)])
        if image_A.mode != 'RGB':
            image_A = to_rgb(image_A)
        if image_B.mode != 'RGB':
            image_B = to_rgb(image_B)
            
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {'A':item_A, 'B':item_B}

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

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
def show_batch(batch, title=""):
    plt.figure(figsize=(10,8))
    plt.imshow(np.transpose(vutils.make_grid(batch, padding=2, normalize=True), (1, 2, 0)))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Saves an image to disk. 
def save_img(batch, path):
    plt.imshow(np.transpose(vutils.make_grid(batch, padding=0, normalize=True), (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path)

# Generates and saves fake images to disk. 
# x is the input to the model, i.e. a noise vector for 
# DCGAN, and an image for CycleGAN. 
# For DCGAN, the noise vector can look as follows: 
# noise_vector = torch.randn(1, args.latent_dim, 1, 1).to(device)
def gen_images(model, num_images, x, path):
    model.eval()
    for i in range(num_images):
        upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        with torch.no_grad():
            out = upsample(model(x[i, :]).detach().cpu())
            save_img(out, path + f"{i}")

# Measures the accuracy. 
def accuracy(target, pred):
    return torch.sum((torch.round(target) == torch.round(pred))) / target.shape[0]