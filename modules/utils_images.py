import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import numpy.fft as fft
import os
import torch.nn.functional as F

############################################################################################################################

def compute_fft_image(img_tensor):
    # img_tensor shape: (C, H, W)
    # Compute FFT magnitude (only on first channel if multiple)
    img_np = img_tensor[0].cpu().numpy()
    fft_img = fft.fftshift(fft.fft2(img_np))
    magnitude = np.log1p(np.abs(fft_img))
    return magnitude

def generate_coordinates(H, W, device):
    x = torch.linspace(-1, 1, steps=H, device=device)
    y = torch.linspace(-1, 1, steps=W, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    return torch.cat((grid_x.reshape(-1,1), grid_y.reshape(-1,1)), dim=1)

def show_image(ax, img_tensor):
    img = img_tensor
    ax.axis('off')

    # Auto denormalize if data is in [-1, 1]
    if img.min() < 0:
        img = img * 0.5 + 0.5

    # Handle channel placement and dimensions
    if img.ndim == 2:
        img_np = img.detach().cpu().numpy()
        ax.imshow(img_np, cmap='gray')

    elif img.ndim == 3:
        if img.shape[0] == 1:  # (1, H, W) → grayscale
            img_np = img[0].detach().cpu().numpy()
            ax.imshow(img_np, cmap='gray')

        elif img.shape[0] == 3:  # (3, H, W) → RGB
            img_np = img.permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(img_np)

        elif img.shape[0] == 5:  # (5, H, W) → show channel 0 using inferno
            img_np = img[0].detach().cpu().numpy()
            ax.imshow(np.flipud(img_np), cmap='RdBu')

        elif img.shape[2] == 1:  # (H, W, 1) → grayscale
            img_np = img[:, :, 0].detach().cpu().numpy()
            ax.imshow(img_np, cmap='gray')

        elif img.shape[2] == 3:  # (H, W, 3) → RGB
            img_np = img.detach().cpu().numpy()
            ax.imshow(img_np)

        else:
            raise ValueError(f"Unsupported 3D tensor shape: {img.shape}")

    else:
        raise ValueError(f"Unsupported tensor shape: {img.shape}")
    

def save_reconstructed_image(tensor, path, n_channels):
    """
    Saves the reconstructed image tensor as a PNG file using PIL.

    Parameters:
    - tensor: torch.Tensor of shape (C, H, W), values in [-1, 1] or [0, 1].
    - path: full file path including .png extension.
    - n_channels: number of image channels (1 for grayscale, 3 for RGB).
    """
    tensor = tensor.detach().cpu()

    # Rescale from [-1, 1] to [0, 1] if necessary
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2

    tensor = tensor.clamp(0, 1)

    if n_channels == 1:
        image_np = (tensor[0] * 255).numpy().astype('uint8')  # shape: (H, W)
        image = Image.fromarray(image_np, mode='L')
    elif n_channels == 3:
        image_np = (tensor.permute(1, 2, 0) * 255).numpy().astype('uint8')  # shape: (H, W, 3)
        image = Image.fromarray(image_np, mode='RGB')
    elif n_channels == 5:  # (5, H, W) → show channel 0 using inferno
        image_np = (tensor.permute(1, 2, 0) * 255).numpy().astype('uint8')  # shape: (H, W, 3)
        image = Image.fromarray(image_np[:,:,0], mode='L')
    else:
        raise ValueError(f"Unsupported number of channels: {n_channels}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def set_target(device, filepath, gray=False):
    ext = os.path.splitext(filepath)[-1].lower()    # get the extension

    ### for loading images
    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']     # Supported image extensions

    
    if ext in image_exts:
        image = Image.open(filepath)
        if image.mode in ['P', 'RGBA']:
            image = image.convert('RGB')

        if gray:
            image = image.convert('L')

        if image.mode == 'L':  # grayscale
            transform = T.Compose([
                T.ToTensor(),           # shape (1, H, W)
                T.Normalize(0.5, 0.5)   # normalize to [-1, 1]
            ])
            img = transform(image).squeeze(0).to(device)  # shape (H, W)
            img = img.unsqueeze(0)                        # shape (1, H, W)
        elif image.mode == 'RGB':
            transform = T.Compose([
                T.ToTensor(),                              # shape (3, H, W)
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            img = transform(image).to(device)             # shape (3, H, W)
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")
    

    ### for loading .bin data
    if ext == '.bin':
        arr = np.fromfile(filepath, dtype=np.float64)
        assert arr.size % 5 == 0, "Data size is not compatible with 5 channels"
        spatial_size = int((arr.size // 5) ** 0.5)
        assert spatial_size * spatial_size * 5 == arr.size, "Data is not square-shaped in space"
        arr = arr.reshape((spatial_size, spatial_size, 5), order='F')
        arr = arr.astype(np.float32)

        print(f'original data size: [5,{spatial_size},{spatial_size}]')

        img = torch.tensor(arr).permute(2, 0, 1).to(device)  # (C, H, W)

        # Normalize each channel independently to [-1, 1]
        img_min = img.amin(dim=(1, 2), keepdim=True)
        img_max = img.amax(dim=(1, 2), keepdim=True)
        img = 2 * (img - img_min) / (img_max - img_min + 1e-8) - 1

        img = F.interpolate(img.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
        img = img.squeeze(0)  # back to (C, 512, 512)
        img = img[:1, :, :]  # keep only the first channel
        
    n_channels = img.shape[0]
    H = img.shape[1]
    W = img.shape[2]
    print(f'n_channels: {n_channels}')
    print(f'img shape: {img.shape}')
    pixel_values = img.permute(1, 2, 0).view(-1, n_channels)

    return img, H, W, pixel_values, n_channels

