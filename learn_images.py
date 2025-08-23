import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

from modules import SIREN, SOS_SIREN, WIRE, ReLU_PE, FINER, GAUSS, SIREN_RFF, SIREN_square, spectral_centroid
from modules import train
from modules import generate_coordinates, set_target, show_image, compute_fft_image, save_reconstructed_image

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

############################################################################################################################
# select your fucking device
if torch.cuda.is_available():
    device = torch.device("cuda")       # NVIDIA GPUs
elif torch.backends.mps.is_available():
    device = torch.device("mps")        # Apple silicon
else:
    device = torch.device("cpu")        # in case you are pathetic

print(f"Using device: {device}")

class config:
    lr = 1e-4               # learning rate
    gamma = 0.998           # learning rate decay factor
    scheculer_step = 20     # iteration frequency to update learning rate
    n_repeats = 1           # number of random trails (to get mean and standard deviation of psnr)
    n_HLs = 5               # number of hidden layers
    nb_epochs = 100       # total training epochs
    HL_dim = 256            # hidden layer dimension

    results_folder = './results/image_fitting/'  # folder to save results
############################################################################################################################
image_files = [
    'training_data/images/camera.png', 
    # 'training_data/images/rock_512.jpg', 
    # 'training_data/images/braided_0064.jpg', 
    # 'training_data/images/castle_512.jpg', 
    # 'training_data/images/general/noise1.png'
               ]
# image_files = ['training_data/images/general/noise1.png', 'training_data/images/camera.png', 'training_data/images/rings.png']
# image_files = ['training_data/images/general/high_freq.png', 'training_data/images/radial.png']
# image_files = ['training_data/images/general/noise2.png', 'training_data/images/general/castle.jpg']
# image_files = ['training_data/images/general/castle_512.jpg']

# Network factory
def get_networks(n_channels, SC=0, S0=0, S1=0, HL_dim=256):
    return [
        SIREN(in_dim=2, HL_dim=HL_dim, out_dim=n_channels, w0=30, first_w0=30, n_HLs=config.n_HLs).to(device),
        SIREN_square(omega_0=30, in_dim=2, HL_dim=HL_dim, out_dim=n_channels, first_omega=30, n_HLs=config.n_HLs, spectral_centeroid = SC, S0=0, S1=0).to(device),
        # FINER(in_features=2, hidden_features=HL_dim, hidden_layers=config.n_HLs-1, out_features=n_channels, first_omega_0=30, hidden_omega_0=30.0).to(device),
        # GAUSS(in_features=2, hidden_features=HL_dim, hidden_layers=config.n_HLs-1, out_features=n_channels, scale=30.0).to(device),
        # WIRE(in_features=2, hidden_features=HL_dim, out_features=n_channels, hidden_layers=config.n_HLs-1).to(device),
        # ReLU_PE(in_features=2,hidden_features=HL_dim, hidden_layers=config.n_HLs-1, out_features=n_channels, outermost_linear=True,
        #         first_omega_0=30, hidden_omega_0=30., scale=10.0, pos_encode=True, sidelength=512, fn_samples=None, use_nyquist=True).to(device),
    ]

os.makedirs("results", exist_ok=True)

############################################################################################################################
for img_file in image_files:
    for repeat in range(config.n_repeats):
        print(f'\nProcessing: {img_file} | Repeat: {repeat+1}')
        base_name = os.path.splitext(os.path.basename(img_file))[0]

        img, H, W, pixel_values, n_channels = set_target(device=device, filepath=img_file, gray=False)
        coords = generate_coordinates(H, W, device)

        SC = spectral_centroid(pixel_values.detach().cpu().numpy())

        # Update number of columns: 1 (GT) + N models + 1 (FFT column)
        n_models = len(get_networks(n_channels, SC=SC, HL_dim=config.HL_dim))
        fig, axes = plt.subplots(2, n_models + 2, figsize=(5 * (n_models + 2), 10))
        fig.subplots_adjust(wspace=0.05)

        # Show GT image and FFT
        show_image(axes[0][0], img)
        axes[0][0].set_title('Ground Truth', fontsize=13)

        fft_gt = compute_fft_image(img)
        axes[1][0].imshow(fft_gt, cmap='plasma')
        axes[1][0].set_title('FFT (GT)', fontsize=13)
        axes[1][0].axis('off')

        fft_errs = []
        model_outputs = []
        psnrs = []

        # Train and collect everything
        for i, model in enumerate(get_networks(n_channels, SC=SC, HL_dim=config.HL_dim)):
            model_name = f"{model.__class__.__name__}"
            batch_size = 256*256 if model_name in ['ReLU_PE', 'WIRE', 'GAUSS'] else 512*512
            psnr, model_output = train(model, coords, pixel_values, config, device, nb_epochs=config.nb_epochs, batch_size=batch_size)

            print(f'peak PSNR ({model_name}): {max(psnr)}')
            reconstructed = model_output.cpu().T.view(n_channels, H, W)
            save_reconstructed_image(reconstructed, f'{config.results_folder}/{base_name}_{model_name}_repeat{repeat+1}.png', n_channels)
            model_outputs.append((model_name, reconstructed, psnr))
            fft_recon = compute_fft_image(reconstructed)
            fft_err = np.abs(fft_recon - fft_gt)
            fft_errs.append(fft_err)
            np.savetxt(f'{config.results_folder}/{base_name}_{model_name}_repeat{repeat+1}.txt', psnr)

        # Compute global vmax after collecting all FFT errors
        global_vmax = np.percentile(np.concatenate([e.flatten() for e in fft_errs]), 85)

        # Plot results
        for i, (model_name, reconstructed, psnr) in enumerate(model_outputs):
            show_image(axes[0][i+1], reconstructed)
            axes[0][i+1].set_title(f'{model_name}, {max(psnr):.2f}', fontsize=13)

            fft_recon = compute_fft_image(reconstructed)
            fft_err = np.abs(fft_recon - fft_gt)
            axes[1][i+1].imshow(fft_err, cmap='Reds', vmin=0, vmax=global_vmax)
            axes[1][i+1].set_title(f'FFT ({model_name})', fontsize=13)
            axes[1][i+1].axis('off')

            axes[1][-1].plot(psnr, label=f'{model_name}')

        # Plot PSNR evolution in last column
        axes[0][-1].axis('off')
        axes[1][-1].set_title('PSNR evolution')
        axes[1][-1].legend()

        plt.tight_layout()
        plt.savefig(f'{config.results_folder}/{base_name}_repeat{repeat+1}.pdf')
        plt.close()

        

