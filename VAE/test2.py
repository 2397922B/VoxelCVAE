import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import zlib
import io
import matplotlib.pyplot as plt
import tarfile


from VAE.VAE2 import VAE
from VAE.VAE2 import ResNetCNN

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
            i=0
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.npy.z'):
                    file = tar.extractfile(member)
                    decompressed_file = zlib.decompress(file.read())
                    with np.load(io.BytesIO(decompressed_file)) as data:
                        #add the name of the npy file to the list
                        self.name.append(member.name)
                        if (data['masks'].shape[0] ==30) and (data['voxel_data'].shape == (64,64,64)):

                        
                            self.voxel_data.append(data['voxel_data'])
                        

                           
                            # Apply occlusions to the original masks (images)
                            original_masks = data['masks']


                            self.masks.append(original_masks)
                 
                            
                        
                        
                        
                      
            
                    
    def __len__(self):
        return len(self.voxel_data)
    
    def __getname__(self):
        return self.name

    def __getitem__(self, idx):
        # Get the 3D voxel data
        voxel_data = torch.from_numpy(self.voxel_data[idx].astype(np.float32))
        voxel_data = voxel_data.unsqueeze(0)  # Add a channel dimension

        # Get all masks for the current sample
        masks = self.masks[idx]
        N = len(masks)
        #add a random shape to the mask
        masks = self.apply_occlusion(masks, 'book', 10, 20, 1)

        # Convert masks to PyTorch tensors
        masks_tensor = [torch.from_numpy(mask.astype(np.float32)) for mask in masks]
        

    

        # Stack all masks into a single tensor
        masks = torch.stack(masks_tensor)

        filename = self.name[idx]

        return voxel_data, masks, filename
    
    def generate_object_mask(self, object_shape, size):
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        
        if object_shape == 'book':
            book_polygon = Polygon([(0, 0), (size, 0), (size, size//2), (size//2, size), (0, size//2)])
            book_polygon = rotate(book_polygon, angle=np.random.randint(0, 360), origin='center')
            x, y = book_polygon.exterior.xy
            draw.polygon(list(zip(x, y)), fill=255)
        elif object_shape == 'laptop':
            laptop_polygon = Polygon([(0, 0), (size, 0), (size, size//2), (0, size//2), (size//4, size//4), (size*3//4, size//4)])
            laptop_polygon = rotate(laptop_polygon, angle=np.random.randint(0, 360), origin='center')
            x, y = laptop_polygon.exterior.xy
            draw.polygon(list(zip(x, y)), fill=255)
        elif object_shape == 'chair':
            chair_line = LineString([(0, 0), (size//2, size//2), (size, 0), (size*3//4, size//4), (size//4, size*3//4), (size//2, size)])
            chair_line = rotate(chair_line, angle=np.random.randint(0, 360), origin=Point(size//2, size//2))
            x, y = chair_line.xy
            draw.line(list(zip(x, y)), fill=255, width=size//10)
        else:
            raise ValueError(f"Unsupported object shape: {object_shape}")
        
        return np.array(mask)
    
    def apply_occlusion(self, masks, object_shape, min_size, max_size, count):
        occluded_masks = []
        
        for mask in masks:
            occluded_mask = mask.copy()
            
            # Find the non-zero pixels in the mask
            non_zero_pixels = np.argwhere(mask > 0)
            
            if len(non_zero_pixels) > 0:
                # Determine the bounding box for the non-zero pixels
                min_y, min_x, *_ = np.min(non_zero_pixels, axis=0)  # Unpack only the first two dimensions
                max_y, max_x, *_ = np.max(non_zero_pixels, axis=0)  # Unpack only the first two dimensions
                
                # Calculate the dimensions of the bounding box
                bbox_height = max_y - min_y + 1
                bbox_width = max_x - min_x + 1
                
                for _ in range(count):
                    # Ensure the occlusion size does not exceed the bounding box dimensions
                    size = min(np.random.randint(min_size, max_size), bbox_height, bbox_width)
                    
                    # Generate random coordinates within the bounding box
                    x = max(min_x, min(max_x - size + 1, mask.shape[1] - size))
                    y = max(min_y, min(max_y - size + 1, mask.shape[0] - size))
                    
                    object_mask = self.generate_object_mask(object_shape, size)
                    occluded_mask[y:y+size, x:x+size] = np.where(object_mask > 0, 0, occluded_mask[y:y+size, x:x+size])
            
            occluded_masks.append(occluded_mask)
        
        return np.array(occluded_masks)
      

        

 
    



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
resnet_cnn = ResNetCNN().to(device)
print(device)


path = '/home/conorbrown/Downloads/VAE/N30.pth'
path2 = '/home/conorbrown/Downloads/VAE/N30_cnn.pth'

model.load_state_dict(torch.load(path))
resnet_cnn.load_state_dict(torch.load(path2))


model.eval()
resnet_cnn.eval()



data_test = NpyTarDataset('/home/conorbrown/Downloads/VAE/datasets/legs_test.tar')
data_train = NpyTarDataset('/home/conorbrown/Downloads/VAE/datasets/legs_train.tar')
test_loader = DataLoader(data_test, batch_size=8, shuffle=False)
train_loader = DataLoader(data_train, batch_size=1, shuffle=False)

train_latents = []
train_ids = []
train_mask_indices = []


with torch.no_grad():
    for train_data, train_mask, file_name in train_loader:
        train_data = train_data.to(device)
        train_mask = train_mask.to(device)
        
        for mask_index in range(30):
            mask = train_mask[:, mask_index, :, :, :]
            mask_latent, var = resnet_cnn(mask)
            sample_latent = model.reparameterize(mask_latent, var)
            train_latents.append(mask_latent.cpu().numpy())
            train_ids.extend(file_name)
            train_mask_indices.extend([mask_index] * train_data.shape[0])

train_latents = np.concatenate(train_latents, axis=0)
print(train_latents.shape)
print(len(train_ids))
print(len(train_mask_indices))

# Perform K-Means Clustering
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(train_latents)

def compute_iou(reconstruction, ground_truth):
    reconstruction = reconstruction.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    ground_truth = ground_truth.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    intersection = (reconstruction * ground_truth).sum(dim=(2, 3, 4))
    union = (reconstruction + ground_truth).sum(dim=(2, 3, 4)) - intersection
    iou = intersection / (union + 1e-7)  # Add a small epsilon to avoid division by zero
    return iou.item()


def render_voxel_grid_with_shading(voxel_array,folder,name):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

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
if not os.path.exists('reconstructions2'):
    os.makedirs('reconstructions2')
if not os.path.exists('reconstructions3'):
    os.makedirs('reconstructions3') 
if not os.path.exists('reconstructions4'):
    os.makedirs('reconstructions4') 
if not os.path.exists('reconstructions5'):
    os.makedirs('reconstructions5')
if not os.path.exists('reconstructions6'):
    os.makedirs('reconstructions6')


total_iou = 0
total_nn_iou = 0
num_samples = 0

with torch.no_grad():
    for model_data, mask, filename in test_loader:
        model_data = model_data.to(device)
        mask = mask.to(device)

        for i in range(model_data.size(0)):  # Iterate over batch samples
            model_data_single = model_data[i].unsqueeze(0)
            mask_single = mask[i].unsqueeze(0)
            current_mask = mask_single[:, 0, :, :, :]
            current_latent, _ = resnet_cnn(current_mask)
            test_latent = current_latent.cpu().numpy()
            original = model_data_single.cpu().numpy()
            original[original > 0.5] = 1
            original[original <= 0.5] = 0
            original1 = original[0, 0, :]

            # Perform Nearest-Neighbor Retrieval
            distances = np.linalg.norm(train_latents - test_latent, axis=1)
            nearest_indices = np.argsort(distances)[:1]
            nearest_ids = [train_ids[idx] for idx in nearest_indices]

            # Retrieve the nearest voxel data from the training dataset
            nearest_voxels = []
            with tarfile.open('/home/conorbrown/Downloads/VAE/datasets/legs_train.tar', 'r') as tar:
                for nearest_id in nearest_ids:
                    member = tar.getmember(nearest_id)
                    file = tar.extractfile(member)
                    decompressed_file = zlib.decompress(file.read())
                    with np.load(io.BytesIO(decompressed_file)) as data:
                        nearest_voxels.append(data['voxel_data'])

            nn_iou = compute_iou(torch.from_numpy(nearest_voxels[0]), torch.from_numpy(original1))
            total_nn_iou += nn_iou

            z_sample = torch.randn(1, 128).to(device)
            combined_latent = torch.cat((z_sample, current_latent), dim=1)
            reconstruction = model.decode(combined_latent)
            reconstruction = reconstruction.cpu().numpy()
            reconstruction[reconstruction > 0.5] = 1
            reconstruction[reconstruction <= 0.5] = 0
            reconstruction1 = reconstruction[0, 0, :]
            iou = compute_iou(torch.from_numpy(reconstruction1), torch.from_numpy(original1))
            total_iou += iou
            num_samples += 1

# Calculate average IoU for the entire test dataset
avg_iou = total_iou / num_samples
avg_nn_iou = total_nn_iou / num_samples

print(f"Average IoU for the entire test dataset: {avg_iou:.4f}")
print(f"Average Nearest-Neighbor IoU for the entire test dataset: {avg_nn_iou:.4f}")


with torch.no_grad():  
    count=0
    
    for model_data, mask, filename in test_loader:
        count=count+1

        if not os.path.exists(f'reconstructions7/{count}'):
            os.makedirs(f'reconstructions7/{count}')
        folder = (f'reconstructions7/{count}')
        text_file_path = os.path.join(folder, 'results.txt')
        model_datab = model_data.to(device)
        
        maskb = mask.to(device)  # mask shape is [B, N+1, 1, 128, 128]
        B, N_plus_1, _, _, _ = maskb.shape
        N = N_plus_1 - 1  # Number of positive/anchor masks per sample

        latent_representations, _ = resnet_cnn(maskb.view(-1, 1, 128, 128))
        latent_representations = latent_representations.view(B, N_plus_1, -1)

        # Reshape the latent representations for t-SNE
        latents_tsne = latent_representations.view(-1, latent_representations.shape[-1]).cpu().numpy()

        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        latents_2d = tsne.fit_transform(latents_tsne)

        # Plot the t-SNE visualization with properly formatted legend
        plt.figure(figsize=(8, 6))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']  # Define colors for each sample
        for i in range(B):
            start_idx = i * (N + 1)
            end_idx = (i + 1) * (N + 1)
            plt.scatter(latents_2d[start_idx:end_idx, 0], latents_2d[start_idx:end_idx, 1], c=colors[i % len(colors)], label=f'Sample {i+1}')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('Latent Space Visualization (t-SNE)')
        plt.legend(title='Samples', loc='upper right', ncol=2)
        plt.tight_layout()
        plt.savefig(folder+"/latent_batch.png")



        # Plot the t-SNE visualization for the first sample in the batch
        plt.figure(figsize=(8, 6))
        first_sample_idx = 0
        start_idx = first_sample_idx * (N + 1)
        end_idx = (first_sample_idx + 1) * (N + 1)
        first_sample_latents = latents_2d[start_idx:end_idx]

        colors = ['g']
        plt.scatter(first_sample_latents[:, 0], first_sample_latents[:, 1], c=colors,alpha=0.5)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('Latent Space Visualization (t-SNE) - First Sample')
        plt.tight_layout()
        plt.savefig(folder+"/latent_first.png")



        difficulty_levels = ['Easy', 'Hard', 'Very Hard']
        centroids = []
        variances = []

        for i in range(3):
            start_idx = i * 10
            end_idx = (i + 1) * 10
            level_latents = first_sample_latents[start_idx:end_idx]
            centroid = np.mean(level_latents, axis=0)
            variance = np.sum((level_latents - centroid) ** 2) / len(level_latents)
            centroids.append(centroid)
            variances.append(variance)

        with open(text_file_path, "w") as text_file:
            # Print the variance for each difficulty level and write to the text file
            for level, variance in zip(difficulty_levels, variances):
                output_str = f"{level} images have a variance of {variance:.4f}"
                print(output_str)
                text_file.write(output_str + "\n")

        # Identify the furthest outlier in the single sample graph
        distances = np.sqrt(np.sum((first_sample_latents - centroids[0]) ** 2, axis=1))
        furthest_outlier_idx = np.argmax(distances)
        furthest_outlier_image = maskb[first_sample_idx, furthest_outlier_idx, 0].cpu().numpy()

        # Save the furthest outlier image
        plt.figure()
        plt.imshow(furthest_outlier_image, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(folder+"/latent_outlier.png")

        unique_batches = np.unique(latents_tsne[:, -1])  # Assuming the last column contains batch information
        closest_samples = []

        for batch in unique_batches:
            batch_mask = latents_tsne[:, -1] == batch
            batch_latents = latents_tsne[batch_mask, :-1]  # Exclude the last column (batch information)
            
            if len(batch_latents) > 0:
                centroid = np.mean(batch_latents, axis=0)
                distances = np.sqrt(np.sum((batch_latents - centroid) ** 2, axis=1))
                closest_idx = np.argmin(distances)
                closest_sample = batch_latents[closest_idx]
                closest_samples.append((batch, closest_sample))

        closest_samples = sorted(closest_samples, key=lambda x: np.sum(x[1]))[:3]

        plt.figure(figsize=(8, 6))
        for i in range(B):
            start_idx = i * (N + 1)
            end_idx = (i + 1) * (N + 1)
            plt.scatter(latents_2d[start_idx:end_idx, 0], latents_2d[start_idx:end_idx, 1], c=colors[i % len(colors)], label=f'Sample {i+1}')
            plt.scatter(first_sample_latents[furthest_outlier_idx, 0], first_sample_latents[furthest_outlier_idx, 1], c='r', label='Furthest Outlier', marker='x', s=100)
            plt.savefig(folder + "/cluster_merge.png")

        for i, (batch, sample) in enumerate(closest_samples):
            sample_idx = np.where((latents_tsne[:, :-1] == sample).all(axis=1))[0][0]
            batch_idx = int(sample_idx // (N + 1))
            mask_idx = sample_idx % (N + 1)
            
            closest_image = maskb[batch_idx, mask_idx, 0].cpu().numpy()
            
            plt.figure()
            plt.imshow(closest_image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(folder + f"/closest_sample_{i+1}.png")
        


        
        
            
        

        #just use the first sample in batch
        model_data = model_data[0].unsqueeze(0).to(device)
        mask = mask[0].unsqueeze(0).to(device)
        print(model_datab.shape)
        print(model_data.shape)

        print(maskb.shape)
        print(mask.shape)
        

        # Pick two distant points in the latent space
        z_point1 = torch.randn(128).to(device)
        z_point2 = torch.randn(128).to(device)

        # Ensure the points are distant by scaling them
        scale_factor = 2.0
        z_point1 = z_point1 * scale_factor
        z_point2 = z_point2 * scale_factor



        z_sample = torch.randn(1, 128).to(device)

        # Use the first sample in the batch for testing
        model_data_single = model_data
        mask_single = mask

        for j in range(10):
        
            # Get the current mask
            current_mask = mask_single[:, i*3, :, :, :]

            # Get the latent representation for the current mask
            current_latent,var = resnet_cnn(current_mask)

            z_cnn = model.reparameterize(current_latent, var)
            combined_latent = torch.cat((z_sample, z_cnn), dim=1)

            # Generate the reconstruction
            reconstruction = model.decode(combined_latent)
            reconstruction = reconstruction.cpu().numpy()

            reconstruction[reconstruction > 0.5] = 1
            reconstruction[reconstruction <= 0.5] = 0
            reconstruction1 = reconstruction[0, 0, :]
            render_voxel_grid_with_shading(reconstruction1, folder, f'mask_{j}_reconstruction')
            #save the mask as an image
            mask = current_mask.cpu().numpy()
            mask = mask[0, 0, :]

            # Define a threshold value (e.g., 50)
            threshold = 3

            # Create a boolean mask where True represents pixels below the threshold
            black_mask = mask < threshold

            # Set all pixels below the threshold to white (255)
            mask[black_mask] = 255

            image = Image.fromarray(mask)
            image = image.convert('RGB')
            image.save(f"{folder}/mask_easy_{j}.png")
            
            original = model_data.cpu().numpy()
            original[original > 0.5] = 1
            original[original <= 0.5] = 0
            original1 = original[0, 0, :]
            render_voxel_grid_with_shading(original1, folder, 'ground_truth')

            current_mask = mask_single[:, 0, :, :, :]
            current_latent,_ = resnet_cnn(current_mask)
            test_latent = current_latent.cpu().numpy()
            test_voxel = original1

        # Perform Nearest-Neighbor Retrieval
        distances = np.linalg.norm(train_latents - test_latent, axis=1)
        nearest_indices = np.argsort(distances)[:60]
        nearest_ids = [train_ids[idx] for idx in nearest_indices]

        # Retrieve the nearest voxel data from the training dataset
        nearest_voxels = []
        nearest_voxel_images = []
        with tarfile.open('/home/conorbrown/Downloads/VAE/datasets/legs_train.tar', 'r') as tar:
            for nearest_id in nearest_ids:
                member = tar.getmember(nearest_id)
                file = tar.extractfile(member)
                decompressed_file = zlib.decompress(file.read())
                with np.load(io.BytesIO(decompressed_file)) as data:
                    nearest_voxels.append(data['voxel_data'])
                    nearest_voxel_images.append(data['masks'][0][0])

        render_voxel_grid_with_shading(nearest_voxels[0], folder, 'nearest_neighbor')
        image = Image.fromarray(nearest_voxel_images[0])
        image = image.convert('RGB')
        image.save(f"{folder}/nearest_neighbor.png")

        nn_iou = compute_iou(torch.from_numpy(nearest_voxels[0]), torch.from_numpy(test_voxel))
        nn_ssim = ssim(nearest_voxels[0], test_voxel, data_range=test_voxel.max() - test_voxel.min())

        # Write the results to the text file
        with open(text_file_path, "a") as text_file:
            text_file.write(f"\nNearest-Neighbor Retrieval - IoU: {nn_iou:.4f}")
            text_file.write(f"\nNearest-Neighbor Retrieval - SSIM: {nn_ssim:.4f}")

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        train_latents_2d = tsne.fit_transform(train_latents)

        # Plot the training latent space in 2D with mean of all training images
        plt.figure(figsize=(10, 8))
        unique_ids = list(set(train_ids))
        colors = plt.cm.get_cmap('viridis', len(unique_ids))
        for i, unique_id in enumerate(unique_ids):
            mask_indices = [idx for idx, train_id in enumerate(train_ids) if train_id == unique_id]
            mask_latents = train_latents_2d[mask_indices]
            plt.scatter(np.mean(mask_latents[:, 0]), np.mean(mask_latents[:, 1]), c=[colors(i)], alpha=0.5, label=unique_id)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('Training Latent Space')
        
        plt.tight_layout()
        plt.savefig(f"{folder}/training_latent_space.png")
        plt.close()


        num_samples = 10

        # Initialize lists to store the metrics
        iou_scores = []
        ssim_scores = []

        anchor = mask_single[:, 0, :, :, :]
        latent,_ = resnet_cnn(anchor)

        for i in range(num_samples):
            z_sample = torch.randn(1, 128).to(device)
            
            # Combine the randomly sampled latent vector with the current mask's latent representation
            combined_latent = torch.cat((z_sample, latent), dim=1)
            
            # Generate the reconstruction
            reconstruction = model.decode(combined_latent)
            reconstruction = reconstruction.cpu().numpy()
            
            reconstruction[reconstruction > 0.6] = 1
            reconstruction[reconstruction <= 0.6] = 0
            reconstruction1 = reconstruction[0, 0, :]
            
            # Compute IoU
            intersection = np.sum(np.logical_and(original1, reconstruction1))
            union = np.sum(np.logical_or(original1, reconstruction1))
            iou = intersection / union
            iou_scores.append(iou)
            
            # Compute SSIM
            ssim_score = ssim(original1, reconstruction1, data_range=reconstruction1.max() - reconstruction1.min())
            ssim_scores.append(ssim_score)
            
            # Save the reconstruction
            render_voxel_grid_with_shading(reconstruction1, folder, f'variation{i}')

        # Compute average metrics
        avg_iou = np.mean(iou_scores)
        avg_ssim = np.mean(ssim_scores)

        # Compute standard deviation
        std_iou = np.std(iou_scores)
        std_ssim = np.std(ssim_scores)

        # Write the results to the text file
        with open(text_file_path, "a") as text_file:
            text_file.write(f"\nAverage IoU: {avg_iou:.4f}")
            text_file.write(f"\nAverage SSIM: {avg_ssim:.4f}")
            text_file.write(f"\nStandard Deviation IoU: {std_iou:.4f}")

        plt.close('all')



                    
                    
                