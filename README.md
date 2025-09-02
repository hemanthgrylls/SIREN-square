# Spectral Bottleneck In Deep Neural Networks: Noise is All You Need

Project website: cfdlabtechnion.github.io/siren_square

Deep neural networks are known to exhibit a spectral learning bias, wherein low-frequency components are learned early in training, while high-frequency modes emerge more gradually in later epochs. However, when the target signal lacks low-frequency components and is dominated by broadband high frequencies, training suffers from a \emph{spectral bottleneck}, and the model fails to reconstruct the entire signal, including the frequency components that lie within its representational capacity. We examine such a scenario in the context of implicit neural representations (INRs) with sinusoidal representation networks (SIRENs), focusing on the challenge of fitting high-frequency-dominant signals that are susceptible to spectral bottleneck. To effectively fit any target signal irrespecitve of it's frequency content, we propose a generalized target-aware \textit{weight perturbation scheme} (WINNER - weight initialization with noise for neural representations) for network initialization. The scheme perturbs uniformly initialized weights with Gaussian noise, where the noise scales are adaptively determined by the spectral centroid of the target signal. We show that the noise scales can provide control over the spectra of network activations and the eigenbasis of the empirical neural tangent kernel. This method not only addresses the spectral bottleneck but also yields faster convergence and with improved representation accuracy, outperforming state-of-the-art approaches in audio fitting and achieving notable gains in image fitting and denoising tasks. Beyond signal reconstruction, our approach opens new directions for adaptive weight initialization strategies in computer vision and scientific machine learning.

---

## Key highlights

- <b>spectral bottleneck</b> can cause INRs to fail representing a signal (image, audio etc.). It is important to incorporate the knowledge of target in to the weight initialization. 
-  A new <b>target-aware weight initialization scheme</b> - WINNER, for implicit neural representations with SIREN is proposed.
-  <b>State-of-the-art 1D audio fitting accuracy</b> for signals dominated by high frequencies.
-  Improved fitting accuracy and faster convergence over baseline SIREN across all target types (audio, images, and 3D shapes).

---

## Repo structure (minimal)
