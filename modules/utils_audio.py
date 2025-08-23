import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf

############################################################################################################################

def set_audio_target(device, file_path, max_samples):
    waveform, sr = wav_to_tensor(file_path)
    waveform /= torch.max(torch.abs(waveform))  # normalize to [-1, 1]
    waveform = waveform.to(device)
    n_samples = min(max_samples, waveform.shape[0])

    return waveform[:n_samples].view(-1,1), n_samples, sr

def wav_to_tensor(filepath, sr=None, mono=True):
    audio, sample_rate = librosa.load(filepath, sr=sr, mono=mono)
    print(f'original signal length: {audio.shape}, sample rate: {sample_rate}')
    tensor = torch.tensor(audio)
    return tensor, sample_rate

def tensor_to_wav(tensor, sample_rate, output_filepath='sample.wav'):
    audio = tensor.to('cpu').detach().numpy()
    sf.write(output_filepath, audio, sample_rate)

def plot_waveforms(waveforms, sample_rates):
    plt.figure(figsize=(3.5, 3))
    
    for idx, (waveform, sr) in enumerate(zip(waveforms, sample_rates)):
        if hasattr(waveform, "cpu"):
            waveform = waveform.cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        time = np.linspace(-1, 1, len(waveform), endpoint=False)
        
        if idx == 0:
            plt.plot(time, waveform, color='black', linewidth=1.0, zorder=2)
        elif idx == 1:
            plt.plot(time, waveform, color='blue', alpha=0.5, linewidth=1.5, zorder=1)

    plt.xlim(0, 0 + 0.001)
    plt.ylim(-1, 1)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    # Set x-ticks to only first and last points
    plt.xticks([0, 0.001], labels=[f"{0}", f"{0.001:.3f}"])
    plt.yticks([-1, 0, 1], labels=[f"{-1}", f"{0}", f"{1}"])

    plt.tight_layout()
    plt.savefig('waveform.pdf')
    plt.show()
    