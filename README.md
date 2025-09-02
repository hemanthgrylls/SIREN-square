# SIREN_square (SIREN²): Target-aware noisy initialization for INRs

SIREN² is a lightweight modification of SIREN that tackles the **spectral bottleneck**: standard INRs learn low frequencies first and may fail to recover high-frequency–dominant targets (e.g., audio) even when representationally capable. We add target-aware Gaussian noise to early weights at initialization, broadening the frequency support and stabilizing optimization. See Fig. 1 (spectral bottleneck) and Sec. 3 for the perturbation scheme. :contentReference[oaicite:0]{index=0}

---

## Method (one paragraph)

Let the first two linear layers of a SIREN be initialized as in Sitzmann et al. Then perturb the weights with zero-mean Gaussian noise only up to layer 2:
\[
W^{(l)} \leftarrow W^{(l)} + \eta^{(l)},\quad 
\eta^{(l)}_{jk}\sim\mathcal{N}\!\left(0,\; (s/\omega_0)^2\right),\quad
l\in\{1,2\},
\]
leaving deeper layers unperturbed. The noise scales \((s_0,s_1)\) are set **target-aware** via the target’s spectral centroid \(\psi\):
\[
\psi = 2\,\frac{\sum_k k\,|\hat y(k)|}{\sum_k |\hat y(k)|},\qquad
s_0 = s^{\max}_0 \Big( 1 - e^{a\sqrt{\psi}/C} \Big),\quad
s_1 = b\,\frac{\psi}{\sqrt{C}},
\]
with recommended \([s^{\max}_0,a,b]=[3500,5,3]\) for audio and \([50,5,0.4]\) for images (see Eq. 11). This widens pre-activation spectra in early layers and slows NTK eigenvalue decay, improving high-frequency receptivity at init; cf. Figs. 7–8. :contentReference[oaicite:1]{index=1}
---

## Results at a glance

- **Audio fitting (150 k samples, 4×222 MLP; 5 runs each):** SIREN² reaches the best PSNR on all clips and the best **average** (≈ 64.5 dB) vs FINER++ (≈ 56.5 dB) and SIREN (≈ 34.8 dB). See Table 1 (page 10) and Fig. 9 (page 11). :contentReference[oaicite:2]{index=2}  
- **Image fitting (4×256):** SIREN² consistently improves PSNR over SIREN; e.g., `noise.png` 36.1 dB vs 21.3 dB (+69%) and `camera.png` 44.9 dB vs 38.9 dB (+15%). See Table 2 (page 11) and Fig. 10. :contentReference[oaicite:3]{index=3}  
- **Denoising:** For 2D fields and images, SIREN² better preserves fine structures (Figs. 11 & 15). Audio denoising uses \((s_0,s_1)\) to control frequency support; Table 3 summarizes PSNR. :contentReference[oaicite:4]{index=4}  
- **Tensor-Train (TT) variant:** Replacing the 4th dense layer with a TT layer (torchtt) reduces params by ~8% while **increasing** PSNR (e.g., `relay.wav` 74.35 dB TT vs 71.73 dB dense). See Suppl. Table 4 (page 18). :contentReference[oaicite:5]{index=5}

---

## Repo structure (minimal)
