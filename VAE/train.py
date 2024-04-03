
import torch
from itertools import chain
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import binary_erosion
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import random
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib as matlibplot


 # Assuming you have this utility for reading the .tar dataset
from VAE import VAE  # Import your PyTorch VAE model her
from VAE import ResNetCNN  # Import your PyTorch ResNetCNN model here


from torch.utils.tensorboard import SummaryWriter


# Constants
learning_rate_1 = 0.0005
learning_rate_2 = 0.00001
momentum = 0.8
batch_size = 10
epoch_num = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
resnet_cnn = ResNetCNN().to(device)
print(device)


#path = '/home/conorbrown/Downloads/VAE/N90.pth'
#path2 = '/home/conorbrown/Downloads/VAE/N90_cnn.pth'
#model.load_state_dict(torch.load(path))
#resnet_cnn.load_state_dict(torch.load(path2))




import tarfile
import numpy as np
import torch
from torch.utils.data import Dataset
import gzip  # For gzip decompression
import io
import zlib
import random
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import scale, rotate, translate

class NpyTarDataset(Dataset):
    def __init__(self, fname):
        self.fname = fname
        self.voxel_data = []  # Initialize voxel_data list
        self.masks = []
        
        with tarfile.open(self.fname, 'r') as tar:
            i = 0
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.npy.z'):
                    file = tar.extractfile(member)
                    decompressed_file = zlib.decompress(file.read())
                    with np.load(io.BytesIO(decompressed_file)) as data:
                        if (data['masks'].shape[0] > 4) and (data['voxel_data'].shape == (64, 64, 64)) :
                            self.voxel_data.append(data['voxel_data'])
                            
                            # Apply occlusions to the original masks (images)
                            original_masks = data['masks']
                        
                            
                            self.masks.append(original_masks)
    
    def __len__(self):
        return len(self.voxel_data)
    
    def __getitem__(self, idx):
        # Get the 3D voxel data
        voxel_data = torch.from_numpy(self.voxel_data[idx].astype(np.float32))
        voxel_data = voxel_data.unsqueeze(0)  # Add a channel dimension
        
        # Get the anchor and positive masks from the current sample
        anchor_idx, positive_idx = random.sample(range(self.masks[idx].shape[0]), 2)
        anchor_mask = torch.from_numpy(self.masks[idx][anchor_idx].astype(np.float32))
        positive_mask = torch.from_numpy(self.masks[idx][positive_idx].astype(np.float32))
        
        # Select a different sample for the negative mask
        all_indices = list(range(len(self.masks)))
        all_indices.remove(idx)  # Exclude the current index
        negative_sample_idx = random.choice(all_indices)
        negative_mask_idx = random.randint(0, self.masks[negative_sample_idx].shape[0] - 1)
        negative_mask = torch.from_numpy(self.masks[negative_sample_idx][negative_mask_idx].astype(np.float32))

        #add a global anchor the 4th mask
        global_mask = torch.from_numpy(self.masks[idx][0].astype(np.float32))
        
        
        # Stack the anchor, positive, and negative masks
        mask = torch.stack([anchor_mask, positive_mask, negative_mask, global_mask], dim=0)
        
        return voxel_data, mask
    








data_train = NpyTarDataset('/home/conorbrown/Downloads/VAE/datasets/legs_train.tar')
data_val = NpyTarDataset('/home/conorbrown/Downloads/VAE/datasets/legs_test.tar')
#data_test = NpyTarDataset('/home/conorbrown/Downloads/VAE/datasets/legs.tar', split='test')
#torch.save(data_train, 'data_train.pth')
#torch.save(data_val, 'data_val.pth')
#torch.save(data_test, 'data_test.pth')
#data_val = torch.load('/home/conorbrown/Downloads/VAE/data_val.pth')
#data_test = torch.load('/home/conorbrown/Downloads/VAE/data_test.pth')
#data_train = torch.load('/home/conorbrown/Downloads/VAE/data_train.pth')

                       

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True,drop_last=True)
val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False,drop_last=True)

print(len(train_loader))
print(len(val_loader))







optimizer = optim.Adam([
    {'params': model.parameters()},
    {'params': resnet_cnn.parameters()}
], lr=learning_rate_1)








writer = SummaryWriter('runs/LEGS2')
with open('CVAE3.txt', 'w') as log_file:

    # Training loop
    
    i = 0
    for epoch in range(epoch_num):
        
        model.train()
        resnet_cnn.train()
            
        
        
        for model_data, mask in train_loader:
            i += 1
           
            
            # Move data and masks to the device
            model_data = model_data.to(device)
            mask = mask.to(device)  # mask shape is [B, 3, 1, 128, 128]

         
            anchor_mask = mask[:, 0, :, :, :]  # Take the first mask as anchor
            positive_mask = mask[:, 1, :, :, :]  # Second mask as positive
            negative_mask = mask[:, 2, :, :, :]  # Third mask as negative
 

            # Get the latent representations from the CNN
            
            anchor_mean = resnet_cnn(anchor_mask)
            positive_latent = resnet_cnn(positive_mask)
   

            recon_batch, mu, logvar = model(model_data,anchor_mean)
        


    
            bce_loss = model.loss(model_data, recon_batch) * 300
            real_kl_loss = model.kl_loss(mu, logvar)
           
            vae_combined_loss = real_kl_loss + bce_loss
            if epoch < 2:
                vae_combined_loss = bce_loss 
           
        
                 
            #t_loss = resnet_cnn.compute_triplet_loss(anchor_latent, positive_latent) * 5
            

            #cnn_combined_loss = t_loss + kl_loss_cnn 
            
            
            optimizer.zero_grad()
            vae_combined_loss.backward()
            optimizer.step()
        

            

            #optimizer_cnn.zero_grad()
            #t_loss.backward()
            #optimizer_cnn.step()
            

            
            if i % 50 == 0:
                #print(f'Epoch {epoch}, Loss: {cnn_combined_loss.item()}\n')
                writer.add_scalar('KL loss', real_kl_loss.item(), i)
                writer.add_scalar('recon loss', bce_loss.item(), i)
                print(f'Epoch {epoch}, Recon Loss: {bce_loss.item()}, KL Loss: {real_kl_loss.item()}\n') 
                #writer.add_scalar('trip loss', t_loss.item(), i)
         
                
    
 
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'NF{epoch}.pth')
            torch.save(resnet_cnn.state_dict(), f'NF{epoch}_cnn.pth')
    

    
    writer.close()
