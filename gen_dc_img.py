###########################################
# File for generating images with the DCGAN
###########################################
import torch 
import torch.nn as nn 
from utils import gen_images
import torch.nn.functional as F 

# Class for storing things such as learning rate, image size...
class Args:
    def __init__(self):
        self.lr = 2e-4
        self.epochs = 1000
        self.b1 = 0.5
        self.b2 = 0.999
        self.latent_dim = 100
        self.img_size = 64
        self.pixels = int(self.img_size ** 2)
        self.channels = 3
        self.img_tuple = (self.channels, self.img_size, self.img_size)
        self.batch_size = 64
        self.g_fmap_size = 64 
        self.d_fmap_size = 64
        self.d_loss_threshold = 0.15
args = Args()

# Defining the DCGAN.
# Source: https://arxiv.org/pdf/1511.06434.pdf
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(args.latent_dim, args.g_fmap_size * 8, 4, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(args.g_fmap_size * 8, args.g_fmap_size * 4, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(args.g_fmap_size * 4, args.g_fmap_size * 2, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(args.g_fmap_size * 2, args.g_fmap_size, 4, 2, 1, bias=False)
        self.conv5 = nn.ConvTranspose2d(args.g_fmap_size, args.channels, 4, 2, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(args.g_fmap_size * 8)
        self.norm2 = nn.BatchNorm2d(args.g_fmap_size * 4)
        self.norm3 = nn.BatchNorm2d(args.g_fmap_size * 2)
        self.norm4 = nn.BatchNorm2d(args.g_fmap_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = x.relu()
        x = self.conv2(x)
        x = self.norm2(x)
        x = x.relu()
        x = self.conv3(x)
        x = self.norm3(x)
        x = x.relu()
        x = self.conv4(x)
        x = self.norm4(x)
        x = x.relu()
        x = self.conv5(x)
        return F.tanh(x)

n_images =  200
dc = torch.load("models/dc_015threshold_1000epochs_label_smoothing", map_location="cpu")
dc.eval()
noise_vector = torch.randn(n_images, 100, 1, 1)
gen_images(dc, noise_vector, "ranking-app/images/dcgan/dc")