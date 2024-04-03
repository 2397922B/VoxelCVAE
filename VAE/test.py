import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import zlib
import io
import matplotlib.pyplot as plt
import tarfile


from VAE import VAE
from VAE import ResNetCNN

from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import rotate
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans









class NpyTarDataset(Dataset):
    def __init__(self, fname):
        self.fname = fname
        self.voxel_data = []  # Initialize voxel_data list
        self.masks = []
        self.name = []
        
        with tarfile.open(self.fname, 'r') as tar:
            i = 0
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.npy.z'):
                    file = tar.extractfile(member)
                    decompressed_file = zlib.decompress(file.read())
                    with np.load(io.BytesIO(decompressed_file)) as data:
                        if (data['masks'].shape[0] ==30) and (data['voxel_data'].shape == (64, 64, 64)) :
                            self.voxel_data.append(data['voxel_data'])
                            self.name.append(member.name)
                            
                            # Apply occlusions to the original masks (images)
                            original_masks = data['masks']
                           
                                
                            
                            self.masks.append(original_masks)
                 
                            
                        
                        
                        
                      
            
                    
    def __len__(self):
        return len(self.voxel_data)
    
    

    def __getitem__(self, idx):
        # Get the 3D voxel data
        voxel_data = torch.from_numpy(self.voxel_data[idx].astype(np.float32))
        voxel_data = voxel_data.unsqueeze(0)  # Add a channel dimension

        # Get all masks for the current sample
        masks = self.masks[idx]
        N = len(masks)
        

        # Convert masks to PyTorch tensors
        masks_tensor = [torch.from_numpy(mask.astype(np.float32)) for mask in masks]
        

    

        # Stack all masks into a single tensor
        masks = torch.stack(masks_tensor)

        filename = self.name[idx]

        return voxel_data, masks, filename
    
    
      

        

 
    



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
resnet_cnn = ResNetCNN().to(device)
print(device)


path = '/home/conorbrown/Downloads/VAE/NF90.pth'
path2 = '/home/conorbrown/Downloads/VAE/NF90_cnn.pth'

model.load_state_dict(torch.load(path))
resnet_cnn.load_state_dict(torch.load(path2))


model.eval()
resnet_cnn.eval()



data_test = NpyTarDataset('/home/conorbrown/Downloads/VAE/datasets/legs_test.tar')
data_train = NpyTarDataset('/home/conorbrown/Downloads/VAE/datasets/legs_train.tar')
test_loader = DataLoader(data_test, batch_size=16, shuffle=False)
train_loader = DataLoader(data_train, batch_size=1, shuffle=False)

train_latents = []
train_ids = []
train_mask_indices = []


with torch.no_grad():
    for train_data, train_mask, file_name in train_loader:
        train_data = train_data.to(device)
        train_mask = train_mask.to(device)
        
        for mask_index in range(1):
            mask = train_mask[:, mask_index, :, :, :]
            mask_latent= resnet_cnn(mask)
           
            train_latents.append(mask_latent.cpu().numpy())
            train_ids.extend(file_name)
            train_mask_indices.extend([mask_index] * train_data.shape[0])

train_latents = np.concatenate(train_latents, axis=0)



def compute_iou(reconstruction, ground_truth):
    intersection = (reconstruction * ground_truth).sum(dim=(1, 2, 3))
    union = (reconstruction + ground_truth).sum(dim=(1, 2, 3)) - intersection
    iou = intersection / (union + 1e-7)  # Add a small epsilon to avoid division by zero
    return iou


def render_voxel_grid_with_shading(voxel_array,folder,name):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    if isinstance(voxel_array, torch.Tensor):
        voxel_array = voxel_array.cpu().numpy()  # Move tensor to CPU and convert to NumPy array
    

    voxels_to_draw = voxel_array > 0 

    
    colors = np.empty(voxel_array.shape + (4,), dtype=np.float32)
  
    colors[voxel_array > 0] = [0.8, 0.9, 1.0, 1]


    ax.voxels(voxels_to_draw, facecolors=colors, edgecolor='k', linewidth=0.2, shade=True)

    
    
    ax.set_xlim(0, voxel_array.shape[0])
    ax.set_ylim(0, voxel_array.shape[1])
    ax.set_zlim(0, voxel_array.shape[2])
    ax.set_box_aspect([1, 1, 1])

    ax.set_facecolor('white')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.view_init(elev=30, azim=30)
    plt.tight_layout()
    plt.savefig(f"{folder}/{name}.png", bbox_inches='tight')
    plt.close(fig)

if not os.path.exists('reconstructions'):
    os.makedirs('reconstructions')
folder = 'reconstructions'

iou_scores_group1 = []
iou_scores_group2 = []
iou_scores_group3 = []
iou_baseline_scores_group1 = []
iou_baseline_scores_group2 = []
iou_baseline_scores_group3 = []


with torch.no_grad():
    first_batch = False
    for j, (voxel_data, images, filenames) in enumerate(test_loader):
        voxel_data = voxel_data.to(device)
        images = images.to(device)
        
        batch_size = voxel_data.shape[0]
        num_masks = images.shape[1]
        
        if first_batch:
            for i in range(batch_size):
                current_mask = images[i, 0, :, :, :]
                current_latent = resnet_cnn(current_mask.unsqueeze(0))
                
                original = voxel_data[i].cpu().numpy()
                original[original > 0.3] = 1
                original[original <= 0.3] = 0
                
                num_reconstructions = 4
                reconstructions = []
                
                for _ in range(num_reconstructions):
                    z_sample = torch.randn(1, 128).to(device)
                    combined_latent = torch.cat((z_sample, current_latent), dim=1)
                    reconstruction = model.decode(combined_latent)
                    reconstruction = reconstruction.cpu().numpy()
                    reconstruction[reconstruction > 0.5] = 1
                    reconstruction[reconstruction <= 0.5] = 0
                    reconstructions.append(reconstruction[0])
                
                # Save the original voxel data, image, and reconstructions
                original_filename = f"{folder}/{filenames[i]}_original.png"
                render_voxel_grid_with_shading(original[0], folder, f"{filenames[i]}_original")
                
                image_filename = f"{folder}/{filenames[i]}_image.png"
                plt.imsave(image_filename, current_mask.cpu().numpy().squeeze(), cmap='gray')
                
                for k, reconstruction in enumerate(reconstructions):
                    reconstruction_filename = f"{folder}/{filenames[i]}_reconstruction_{k+1}.png"
                    render_voxel_grid_with_shading(reconstruction[0], folder, f"{filenames[i]}_reconstruction_{k+1}")
            
            first_batch = False
        
        for j in range(num_masks):
            current_masks = images[:, j, :, :, :]
            current_latents = resnet_cnn(current_masks)
            
            original = voxel_data.cpu().numpy()
            original[original > 0.3] = 1
            original[original <= 0.3] = 0
            
            distances = np.linalg.norm(train_latents - current_latents.cpu().numpy()[:, np.newaxis, :], axis=2)
            nearest_indices = np.argmin(distances, axis=1)
            nearest_ids = [train_ids[idx] for idx in nearest_indices]
            
            # Load the voxel data of the nearest neighbor images for the current batch
            nearest_voxels = []
            with tarfile.open('/home/conorbrown/Downloads/VAE/datasets/legs_train.tar', 'r') as tar:
                for nearest_id in nearest_ids:
                    member = tar.getmember(nearest_id)
                    file = tar.extractfile(member)
                    decompressed_file = zlib.decompress(file.read())
                    with np.load(io.BytesIO(decompressed_file)) as data:
                        nearest_voxels.append(data['voxel_data'])
            
            nearest_voxels = torch.from_numpy(np.stack(nearest_voxels)).to(device)
            
            z_sample = torch.randn(batch_size, 128).to(device)
            combined_latents = torch.cat((z_sample, current_latents), dim=1)
            reconstructions = model.decode(combined_latents)
            reconstructions = reconstructions.cpu().numpy()
            reconstructions[reconstructions > 0.5] = 1
            reconstructions[reconstructions <= 0.5] = 0
            
            iou_batch = [compute_iou(torch.from_numpy(rec), torch.from_numpy(orig)).item() for rec, orig in zip(reconstructions, original)]
            iou_baseline_batch = [compute_iou(torch.from_numpy(nearest), torch.from_numpy(orig)).item() for nearest, orig in zip(nearest_voxels.cpu().numpy(), original)]
            for k, (iou, filename) in enumerate(zip(iou_batch, filenames)):
                if iou_batch > iou_baseline_batch:
                    # Save ground truth voxel data
                    ground_truth_filename = f"{folder}/{filename}_ground_truth.png"
                    print(original[k].shape)
                    render_voxel_grid_with_shading(original[k][0], folder, f"{filename}_ground_truth")
                    
                    # Save reconstruction voxel data
                    reconstruction_filename = f"{folder}/{filename}_reconstruction.png"
                    render_voxel_grid_with_shading(reconstructions[k][0], folder, f"{filename}_reconstruction")
                    
                    # Save image
                    image_filename = f"{folder}/{filename}_image.png"
                    plt.imsave(image_filename, current_masks[k].cpu().numpy().squeeze(), cmap='gray')

                    #retrieval
                    nearest_filename = f"{folder}/{filename}_nearest.png"
                    render_voxel_grid_with_shading(nearest_voxels.cpu().numpy()[k], folder, f"{filename}_nearest")
            if j < 10:
                iou_scores_group1.extend(iou_batch)
                iou_baseline_scores_group1.extend(iou_baseline_batch)
            elif j < 20:
                iou_scores_group2.extend(iou_batch)
                iou_baseline_scores_group2.extend(iou_baseline_batch)
            else:
                iou_scores_group3.extend(iou_batch)
                iou_baseline_scores_group3.extend(iou_baseline_batch)

print("Group 1 (First 10 images):")
print(f"Mean IoU: {np.mean(iou_scores_group1):.4f}")
print(f"Mean IoU (baseline): {np.mean(iou_baseline_scores_group1):.4f}")

print("Group 2 (Second 10 images):")
print(f"Mean IoU: {np.mean(iou_scores_group2):.4f}")
print(f"Mean IoU (baseline): {np.mean(iou_baseline_scores_group2):.4f}")

print("Group 3 (Third 10 images):")
print(f"Mean IoU: {np.mean(iou_scores_group3):.4f}")
print(f"Mean IoU (baseline): {np.mean(iou_baseline_scores_group3):.4f}")