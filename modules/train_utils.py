import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

############################################################################################################################

def batched_forward(model, coords, batch_size=65536):
    """
    Perform <<model(coords_batch)>> in smaller batches to avoid OOM.
    """
    outputs = []
    N = coords.shape[0]
    for i in range(0, N, batch_size):
        coords_batch = coords[i:i+batch_size]
        out = model(coords_batch)
        outputs.append(out.detach().cpu())
    return torch.cat(outputs, dim=0)


def train(model, coords, ground_truth, config, device, nb_epochs, batch_size):
    model.to(device)
    coords = coords.to(device)
    ground_truth = ground_truth.to(device)

    loss_fun = nn.MSELoss()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(model_optimizer, step_size=config.scheculer_step, gamma=config.gamma)

    psnr_history = []
    total_points = coords.shape[0]
    best_psnr = -float('inf')
    best_model_state = None

    pbar = tqdm(range(nb_epochs))
    for _ in pbar:
        indices = torch.randperm(total_points)
        epoch_loss = 0.0

        for i in range(0, total_points, batch_size):
            batch_indices = indices[i:i + batch_size]
            coords_batch = coords[batch_indices, :]
            gt_batch = ground_truth[batch_indices, :]

            output = model(coords_batch)
            loss = ((output - gt_batch) ** 2).mean()
            epoch_psnr = 20 * np.log10(1.0 / np.sqrt(loss.item()))

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            epoch_loss += loss.item()

        # Full-batch PSNR estimate (optional but used in your original logic)
        epoch_psnr = 20 * np.log10(1.0 / np.sqrt(loss.item()))
        psnr_history.append(epoch_psnr)

        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            best_model_state = model.state_dict()

        pbar.set_postfix({'PSNR': f'{epoch_psnr:.2f} dB'})
        scheduler.step()

    # Restore best model and compute its output
    model.load_state_dict(best_model_state)
    output = batched_forward(model, coords, batch_size=65536)

    return psnr_history, output

