##############################################
# File for generating images with the CycleGan
##############################################
import torch 
import torch.nn as nn 
from utils import gen_images, CycleDataset, save_img
import torch.nn.functional as F 
from torchvision import transforms
from PIL import Image

# Class for storing things such as learning rate, image size...
class Args:
    def __init__(self):
        self.lr = 2e-4
        self.epochs = 200
        self.b1 = 0.5
        self.b2 = 0.999
        self.img_size = 64
        self.pixels = int(self.img_size ** 2)
        self.channels = 3
        self.img_tuple = (self.channels, self.img_size, self.img_size)
        self.batch_size = 1
        self.d_loss_threshold = 0.5
        self.n_res_blocks = 9
        self.decay_epoch = 5
        self.start_epoch = 1
args = Args()

# Defining the Generator. 
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_features, in_features, 3)
        self.conv2 = nn.Conv2d(in_features, in_features, 3)
        self.norm1 = nn.InstanceNorm2d(in_features)
        self.norm2 = nn.InstanceNorm2d(in_features)
    
    def block(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = x.relu()
        x = self.pad2(x)
        x = self.conv2(x)
        return self.norm2(x)
    
    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_features = 64
        self.in_features = self.out_features
        self.__compile_model()
        
    # First convolutional block. 
    def __conv1(self):
        self.model += [
            nn.ReflectionPad2d(args.channels),
            nn.Conv2d(args.channels, self.out_features, 7),
            nn.InstanceNorm2d(self.out_features),
            nn.ReLU(inplace=True)
        ]
                       
    # Downsampling
    def __downsample(self):
        for _ in range(2):
            self.out_features *= 2  
            self.model += [
                nn.Conv2d(self.in_features, self.out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(self.out_features),
                nn.ReLU(inplace=True)
            ]
            self.in_features = self.out_features

    def __residual_blocks(self):
        for _ in range(args.n_res_blocks):
            self.model += [ResidualBlock(self.out_features)]

    # Upsampling 
    def __upsample(self):
        for _ in range(2):
            self.out_features //= 2
            self.model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.in_features, self.out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            self.in_features = self.out_features

    # Output layer
    def __output(self):
        self.model += [
            nn.ReflectionPad2d(args.channels),
            nn.Conv2d(self.out_features, args.channels, 7),
            nn.Tanh()
        ]
    
    # Compiles the model
    def __compile_model(self):
        self.model = []
        self.__conv1()
        self.__downsample()
        self.__residual_blocks()
        self.__upsample()
        self.__output()
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x)
    
# Loading the images.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

n_images =  20
loader = torch.utils.data.DataLoader(
    CycleDataset("data/", transform=transform, unaligned=True),
    batch_size=n_images,
    shuffle=True
)

cycle_ba = torch.load("models/cycle_ba_200epochs", map_location="cpu")
cycle_ab = torch.load("models/cycle_ab_200epochs", map_location="cpu")
cycle_ba.eval()
cycle_ab.eval()
x = next(iter(loader))['B']
out = cycle_ba(x).detach().cpu()
gen_images(cycle_ba, x, "example_images/fake_monet/fake_monet")
gen_images(cycle_ab, out, "example_images/fake_photos/fake_photo" )
for i, img in enumerate(x):
    save_img(img, "example_images/photos/real_photo" + f"{i}")