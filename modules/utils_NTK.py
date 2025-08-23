import numpy as np
import torch

############################################################################################################################

def compute_ntk_fourier_spectrum_with_fft(model, n_samples, inputs, device="cpu"):
    # Move model to device
    model.to(device)

    # Compute gradients for NTK
    gradients = []
    for input_batch in inputs:
        model.zero_grad()
        output = model(input_batch.unsqueeze(0))
        output.backward()
        grads = torch.cat([param.grad.view(-1) for param in model.parameters()])
        gradients.append(grads)

    # Stack gradients and calculate NTK matrix
    gradients = torch.stack(gradients).to(device)
    ntk_matrix = torch.matmul(gradients, gradients.T) / (n_samples**2)
    ntk_matrix_np = ntk_matrix.cpu().numpy()

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(ntk_matrix_np)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Compute weighted FFT magnitude spectrum from eigenvectors
    spectrum = np.zeros_like(np.fft.fftshift(np.fft.fft(eigenvectors[:, 0])), dtype=np.float64)
    for i in range(len(sorted_eigenvalues)):
        vec = eigenvectors[:, i]
        fft_vec = np.fft.fftshift(np.fft.fft(vec))
        power_spectrum = np.abs(fft_vec) ** 2
        spectrum += sorted_eigenvalues[i] * power_spectrum
    spectrum /= n_samples

    # Normalized x-axis and FFT frequency axis
    eig_idxs = np.linspace(0, len(sorted_eigenvalues), len(sorted_eigenvalues))
    eig_idxs= eig_idxs.astype(int)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_samples)) * 2 * np.pi

    return {
        "ntk_matrix": ntk_matrix_np,
        "eig_idxs": eig_idxs,
        "sorted_eigenvalues": sorted_eigenvalues,
        "freqs": freqs,
        "fft_spectrum": spectrum
    }


def compute_ntk(model, n_samples, inputs, device="cpu"):
    # Move model to device
    model.to(device)

    # Compute gradients for NTK
    gradients = []
    for input_batch in inputs:
        model.zero_grad()
        output = model(input_batch.unsqueeze(0))
        output.backward()
        grads = torch.cat([param.grad.view(-1) for param in model.parameters()])
        gradients.append(grads)

    # Stack gradients and calculate NTK matrix
    gradients = torch.stack(gradients).to(device)
    ntk_matrix = torch.matmul(gradients, gradients.T) / (n_samples**2)
    ntk_matrix_np = ntk_matrix.cpu().numpy()

    return ntk_matrix_np