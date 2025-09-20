import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from modules import SIREN, WIRE, ReLU_PE, FINER, GAUSS, SIREN_RFF, SIREN_square, spectral_centroid
from modules import train
from modules import set_audio_target, wav_to_tensor, tensor_to_wav, plot_waveforms

# matplotlib.use('Agg')  # Use non-interactive backend

os.environ["MPLBACKEND"] = "Agg"
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
    gamma = 0.99            # learning rate decay factor
    scheculer_step = 20     # iteration frequency to update learning rate
    n_repeats = 1           # number of random trails (to get mean and standard deviation of psnr)
    n_HLs = 4               # number of hidden layers
    nb_epochs = 1000        # total training epochs
    HL_dim = 222            # hidden layer dimension
    max_samples = 150000
    n_channels = 1
    results_folder = './results/audio_fitting/'

############################################################################################################################
files = ['training_data/audio/tetris.wav',
         'training_data/audio/relay.wav',
         'training_data/audio/arch.wav',
         'training_data/audio/tap.wav',
         'training_data/audio/whoosh.wav',
         'training_data/audio/voltage.wav',
         'training_data/audio/foley.wav',
         'training_data/audio/shattered.wav',
         'training_data/audio/radiation.wav',
         'training_data/audio/sparking.wav',
         'training_data/audio/birds.wav',
         'training_data/audio/gt_bach.wav'
         ]

def get_networks(config, n_channels, HL_dim=256, SC=0, S0=0, S1=0):
    return [
        SIREN(in_dim=1, HL_dim=HL_dim, out_dim=n_channels, first_w0=3000, w0=30, n_HLs=config.n_HLs).to(device),
        SIREN_square(omega_0=30, in_dim=1, HL_dim=HL_dim, out_dim=1, first_omega=30, n_HLs=config.n_HLs, spectral_centeroid = SC, S0=S0, S1=S1).to(device),
        # FINER(in_features=1, hidden_features=HL_dim, hidden_layers=config.n_HLs-1, out_features=n_channels, first_omega_0=30, hidden_omega_0=30.0).to(device),
        # GAUSS(in_features=1, hidden_features=HL_dim, hidden_layers=config.n_HLs-1, out_features=n_channels, scale=30.0).to(device),
        # WIRE(in_features=1, hidden_features=HL_dim, out_features=n_channels, hidden_layers=config.n_HLs-1).to(device),
        # ReLU_PE(in_features=1,hidden_features=HL_dim, hidden_layers=config.n_HLs-1, out_features=n_channels, outermost_linear=True,
        #         first_omega_0=30, hidden_omega_0=30., scale=10.0, pos_encode=True, sidelength=512, fn_samples=None, use_nyquist=True).to(device),
    ]

os.makedirs("results", exist_ok=True)

############################################################################################################################
for file in files:
    for repeat in range(config.n_repeats):
        print(f'\nProcessing: {file} | Repeat: {repeat+1}')
        base_name = os.path.splitext(os.path.basename(file))[0]

        waveform_gt, n_samples, sample_rate = set_audio_target(device, file, config.max_samples)
        coords = torch.linspace(-1, 1, n_samples, device=device).view(-1,config.n_channels)

        print(waveform_gt.detach().cpu().numpy().shape)

        SC = spectral_centroid(waveform_gt.detach().cpu().numpy())

        for i, model in enumerate(get_networks(config, n_channels=config.n_channels, HL_dim=config.HL_dim, SC=SC, S0=3000, S1=1)):
            model_name = f"{model.__class__.__name__}"

            if model_name == 'ReLU_PE' or model_name == 'WIRE' or model_name == 'FINER' or model_name == 'GAUSS':
                batch_size = min(n_samples,256*256)
            else:
                batch_size = min(n_samples,512*512)

            psnr, model_output = train(model, coords, waveform_gt, config, device, nb_epochs=config.nb_epochs, batch_size=batch_size)
            
            print(f'peak PSNR ({model_name}): {max(psnr)}')
            np.savetxt(f'{config.results_folder}/{base_name}_{model_name}_repeat{repeat+1}.txt', psnr)
            tensor_to_wav(model_output[:, 0], sample_rate, f'{config.results_folder}/{base_name}_{model_name}_repeat{repeat+1}.wav')
            tensor_to_wav(waveform_gt, sample_rate, f'{config.results_folder}/{base_name}_gt.wav')

