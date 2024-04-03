


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetCNN(nn.Module):
    def __init__(self):
        super(ResNetCNN, self).__init__()
        
        # Original VGG with BatchNorm
        vgg16_bn = models.vgg16_bn(pretrained=True)
        features = list(vgg16_bn.features.children())
        
        # Modify the first layer to accept 1 channel input
        features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        # Use VGG up to layer 27 for feature extraction
        self.vgg = nn.Sequential(*features[:27])
        
        # Additional layers to reduce features to 64
        self.reduction_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 64, kernel_size=3, padding=1), # Output size is now 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Single output layer instead of separate mean and variance layers
        self.output_layer = nn.Linear(64, 64) # Adjusted the output size to 64
    
    def forward(self, x):
        # Feature extraction and reduction
        x = self.vgg(x)
        x = self.reduction_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        
        # Single point output
        output = self.output_layer(x)
        
        return output
    
    def compute_triplet_loss(self, anchor_output, positive_output, negative_output, margin=1.0):
        # Compute the Euclidean distance between anchor and positive
        positive_distance = torch.nn.functional.pairwise_distance(anchor_output, positive_output)

        # Compute the Euclidean distance between anchor and negative
        negative_distance = torch.nn.functional.pairwise_distance(anchor_output, negative_output)

        # Compute triplet loss
        losses = torch.relu(positive_distance - negative_distance + margin)

        # Compute mean loss over the batch
        loss = torch.mean(losses)

        return loss

        

class VAE(nn.Module):
    def __init__(self,z_dim = 128 ,*args, **kwargs):
        super(VAE, self).__init__(*args, **kwargs) 

        self.enc_conv1 = nn.Sequential(nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=0), nn.BatchNorm3d(8), nn.ELU())
        self.enc_conv2 = nn.Sequential(nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm3d(16), nn.ELU())
        self.enc_conv3 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0), nn.BatchNorm3d(32), nn.ELU())
        self.enc_conv4 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm3d(64), nn.ELU())
        self.enc_conv5 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm3d(128), nn.ELU())

        #  Adjusted fully connected layers to include condition_dim
        self.fc1 = nn.Linear(128 * 8 * 8 * 8, 512) 
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_sigma = nn.Linear(512, z_dim)
        self.bn_mu = nn.BatchNorm1d(z_dim)
        self.bn_logvar = nn.BatchNorm1d(z_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(128 + 64, 128 * 8 * 8 * 8)
        self.dec_unflatten = nn.Unflatten(1, (128, 8, 8, 8))
        self.dec_conv1 = nn.Sequential(nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(64), nn.ELU())
        self.dec_conv2 = nn.Sequential(nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(32), nn.ELU())
        self.dec_conv3 = nn.Sequential(nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(16), nn.ELU())
        self.dec_conv4 = nn.Sequential(nn.ConvTranspose3d(16, 8, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(8), nn.ELU())  # Output: (8, 64, 64, 64)
        self.dec_conv5 = nn.Sequential(nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(1), nn.Sigmoid())  # Output: (1, 64, 64, 64)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        x = self.enc_conv4(x)
        x = self.enc_conv5(x)
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        mu = self.bn_mu(self.fc_mu(x))
        logvar = self.bn_logvar(self.fc_sigma(x))
        return mu, logvar

    def decode(self, z):

        
        z = F.elu(self.dec_fc1(z))

        z = self.dec_unflatten(z)
        
        z = self.dec_conv1(z)
     
        z = self.dec_conv2(z)
     
        z = self.dec_conv3(z)
        
        z = self.dec_conv4(z)

        z = self.dec_conv5(z)
       
        return z

    def forward(self, modeldata,latent_cnn):


        mu_voxel, logvar_voxel = self.encode(modeldata)
        z_vae = self.reparameterize(mu_voxel, logvar_voxel)
        combined_latent = torch.cat((z_vae, latent_cnn), dim=1)
        reconstructed = self.decode(combined_latent)
        return reconstructed, mu_voxel, logvar_voxel

    def loss(self, inputs, outputs):
        
        outputs_clip = torch.clip(outputs, 1e-7, 1.0 - 1e-7)
        loss = -(98.0 * inputs * torch.log(outputs_clip) +  2 * (1.0 - inputs) * torch.log(1.0 - outputs_clip))
        return loss.mean()
    
    def focal_loss(self, inputs, outputs, alpha=0.25, gamma=2):
        outputs_clip = torch.clip(outputs, 1e-7, 1.0 - 1e-7)
        
        # Compute the focal loss
        pt = torch.where(inputs == 1, outputs_clip, 1 - outputs_clip)
        focal_weight = (1 - pt) ** gamma
        ce_loss = -torch.log(pt)
        focal_loss = alpha * focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def loss2(self, inputs, outputs):
        
        outputs_clip = torch.clip(outputs, 1e-7, 1.0 - 1e-7)
        loss = -(95.0 * inputs * torch.log(outputs_clip) +  5 * (1.0 - inputs) * torch.log(1.0 - outputs_clip))
        return loss.mean()
    
    #this loss compares the number of voxels occupied in the input and output
    def occupancy_loss(self, inputs, outputs):
        loss = torch.sum(inputs) - torch.sum(outputs)
        return loss.mean()

    def kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        return kl_loss
    

def initialize_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

# Model initialization
vae = VAE()
resnet_cnn = ResNetCNN()

# Apply weight initialization
vae.apply(initialize_weights)