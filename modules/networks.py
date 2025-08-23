import torch
import torch.nn as nn
import numpy as np
import math
import sys
from collections import OrderedDict
from scipy import signal
import matplotlib.pyplot as plt
import torch.nn.init as init

@torch.jit.script
def sine_block(x: torch.Tensor, w0: float, a0: float):
    return a0 * torch.sin(w0 * x)

@torch.jit.script
def gaussian_core(x: torch.Tensor, scale: float):
    return torch.exp(-(scale * x)**2)

class SineLayer(nn.Module):
    def __init__(self, w0, A=1.0):
        super().__init__()
        self.a0 = A
        self.w0 = w0

    def forward(self,x):
        return sine_block(x, self.w0, self.a0)
    
class FinerLayerH(nn.Module):
    def __init__(self, w0, A=1.0):
        super().__init__()
        self.a0 = A
        self.w0 = w0

    def forward(self,x):
        with torch.no_grad():
            scale = torch.abs(x) + 1
        return sine_block(scale * x, self.w0, self.a0)


def spectral_centroid(x):
    # Last axis is always channels
    num_channels = x.shape[-1]
    centroids = []

    for c in range(num_channels):
        data_c = x[..., c]  # Extract channel c
        ndim = data_c.ndim

        if ndim == 1:  # Audio single signal
            spectrum = np.abs(np.fft.rfft(data_c))
            freq_bins = np.fft.rfftfreq(data_c.shape[-1], d=1)
            weighted_sum = np.sum(spectrum * freq_bins)
            sum_of_weights = np.sum(spectrum)
            centroid = weighted_sum / sum_of_weights if sum_of_weights != 0 else 0

        elif ndim == 2:  # Image (H, W)
            spectrum = np.abs(np.fft.rfft(data_c, axis=1))
            freq_bins = np.fft.rfftfreq(data_c.shape[1], d=1)
            weighted_sum = np.sum(spectrum * freq_bins, axis=1)
            sum_of_weights = np.sum(spectrum, axis=1)
            row_centroids = np.divide(weighted_sum, sum_of_weights, out=np.zeros_like(weighted_sum), where=sum_of_weights != 0)
            centroid = np.mean(row_centroids)

        elif ndim == 3:  # 3D data (H, W, D)
            spectrum = np.abs(np.fft.rfft(data_c, axis=2))
            freq_bins = np.fft.rfftfreq(data_c.shape[2], d=1)
            weighted_sum = np.sum(spectrum * freq_bins, axis=2)
            sum_of_weights = np.sum(spectrum, axis=2)
            slice_centroids = np.divide(weighted_sum, sum_of_weights, out=np.zeros_like(weighted_sum), where=sum_of_weights != 0)
            centroid = np.mean(slice_centroids)

        else:
            raise ValueError("Unsupported data dimensionality after removing channel axis.")

        centroids.append(centroid)

    # Average across channels and normalize to [0, 1]
    return (np.mean(centroids) * 2) / num_channels


############################################################################################################################
# simplified efficient implementation of SIREN
class SIREN(nn.Module):
    def __init__(self, in_dim, HL_dim, out_dim, w0=30, first_w0=3000, n_HLs=4):
        super().__init__()
        self.net = []
        
        self.net.append(nn.Linear(in_dim, HL_dim))
        self.net.append(SineLayer(first_w0))
        for _ in range(n_HLs-1):
            self.net.append(nn.Linear(HL_dim, HL_dim))
            self.net.append(SineLayer(w0))
        self.net.append(nn.Linear(HL_dim, out_dim))
        
        self.net = nn.Sequential(*self.net)

        # init weights
        with torch.no_grad():
            self.net[0].weight.uniform_(-1.0/in_dim, 1.0/in_dim)
            for i in range(n_HLs):
                self.net[(i+1)*2].weight.uniform_(-np.sqrt(6.0/HL_dim)/w0, np.sqrt(6.0/HL_dim)/w0)

    def forward(self, x):
        return self.net(x)

############################################################################################################################
class SIREN_square(nn.Module):
    def __init__(self, omega_0=30, in_dim=1, HL_dim=256, out_dim=1, first_omega=30, n_HLs=4, spectral_centeroid = 0, S0=0, S1=0):
        super().__init__()
        self.in_dim = in_dim
        self.omega_0 = omega_0
        self.S0 = S0    # noise scale between INPUT and 1st HIDDEN LAYER
        self.S1 = S1    # noise scale between 1st and 2nd HIDDEN LAYERS
        self.SC = spectral_centeroid
        self.n_ch = out_dim

        # define network architecture
        self.net = []
        self.net.append(nn.Linear(in_dim, HL_dim))
        self.net.append(SineLayer(first_omega))
        for _ in range(n_HLs-1):
            self.net.append(nn.Linear(HL_dim, HL_dim))
            self.net.append(SineLayer(omega_0))
        self.net.append(nn.Linear(HL_dim, out_dim))
        
        self.net = nn.Sequential(*self.net)

        # initialize weights
        with torch.no_grad():
            self.net[0].weight.uniform_(-1.0/in_dim, 1.0/in_dim)

            for i in range(n_HLs):
                self.net[(i+1)*2].weight.uniform_(-np.sqrt(6.0/HL_dim)/self.omega_0, np.sqrt(6.0/HL_dim)/self.omega_0)

            self.weights0 = self.net[0].weight.detach().clone()
            self.weights2 = self.net[2].weight.detach().clone()
        
        # add noise
        self.set_noise_scales()
        self.add_noise()

    def set_noise_scales(self):
        # insert the emperical formula to set S0 and S1 values as a function of (SC/n_ch)
        if self.in_dim == 1:    # audio
            a, b = 7, 3
            self.S0 = 3500*(1-np.exp(-a*self.SC/self.n_ch))
            self.S1 = self.SC/self.n_ch * b
        elif self.in_dim == 2:  # images
            a, b = 5, 0.4
            self.S0 = 50*(1-np.exp(-a*self.SC/self.n_ch))
            self.S1 = self.SC/self.n_ch * b
        elif self.in_dim == 3:  # 3D
            self.S0 = self.S0
            self.S1 = self.S1
        
        # or manually set the noise scales
        # self.S0 = self.S0
        # self.S1 = self.S1

        print(f'spectral centeroid = {self.SC}, SIREN^2 set to noise scales S0={self.S0} and S1={self.S1}')

    def add_noise(self):
        with torch.no_grad():
            # INPUT LAYER --> FIRST HIDDEN LAYER (uniform + noise)
            scale = self.S0 / self.omega_0
            self.net[0].weight.copy_(self.weights0 + torch.randn_like(self.weights0) * scale)

            # FIRST --> SECOND HIDDEN LAYER (uniform + noise)
            scale = self.S1 / self.omega_0
            self.net[2].weight.copy_(self.weights2 + torch.randn_like(self.weights2) * scale)

            # Free memory after use
            del self.weights0, self.weights2
            torch.cuda.empty_cache()

    def forward(self, coords):
        return self.net(coords)

############################################################################################################################
class RFF(nn.Module):
    def __init__(self, in_dim, n_rff_features):
        super(RFF, self).__init__()
        B = torch.normal(mean = 0.0, std = 30.0, size=(in_dim, n_rff_features))
        self.register_buffer('B', B)

    def forward(self, x):
        x_map = torch.matmul(x, self.B)
        return torch.cat([torch.cos(x_map), torch.sin(x_map)], dim=-1)

class SIREN_RFF(nn.Module):
    def __init__(self, in_dim, HL_dim, out_dim, w0=30, first_w0=3000, n_HLs=4):
        super().__init__()
        self.n_rff_features = HL_dim//2
        self.first_w0 = first_w0
        self.net = []

        # define the net
        self.net.append(RFF(in_dim, self.n_rff_features))
        self.net.append(nn.Linear(2*self.n_rff_features, HL_dim))
        self.net.append(SineLayer(first_w0))
        for _ in range(n_HLs-1):
            self.net.append(nn.Linear(HL_dim, HL_dim))
            self.net.append(SineLayer(w0)) 
        self.net.append(nn.Linear(HL_dim, out_dim))
        
        self.net = nn.Sequential(*self.net)

        # init weights
        with torch.no_grad():
            self.net[1].weight.uniform_(-1.0/in_dim, 1.0/in_dim)
        for i in range(n_HLs):
            with torch.no_grad():
                self.net[(i+1)*2+1].weight.uniform_(-np.sqrt(6.0/HL_dim)/w0, np.sqrt(6.0/HL_dim)/w0)

    def forward(self,x):
        return self.net(x)


############################################################################################################################
# FINER, ref:https://github.com/liuzhen0212/FINER.git
class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale != None:
            self.init_first_bias()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
                # print('init fbs', self.first_bias_scale)

    def generate_scale(self, x):
        if self.scale_req_grad: 
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
        
    def forward(self, input):
        x = self.linear(input)
        scale = self.generate_scale(x)
        out = torch.sin(self.omega_0 * scale * x)
        return out

class FINER(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.0, bias=True, 
                 first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.out_features = out_features
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output.view(-1,self.out_features)


############################################################################################################################
class FINER_plus_plus(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.0, bias=True, 
                 first_bias_scale=5, scale_req_grad=False):
        super().__init__()
        self.out_features = out_features
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output.view(-1,self.out_features)
    

############################################################################################################################
## WIRE Ref：https://github.com/vishwa91/wire
class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    
class WIRE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,
                 hidden_layers=4, first_omega_0=10, hidden_omega_0=10., scale=10.0):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        hidden_features = int(hidden_features/np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'    
        
        # Legacy parameter
        self.pos_encode = False
        
        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        output = self.net(x)
        
        if self.wavelet == 'gabor':
            return output.real
         
        return output


############################################################################################################################
class ReLULayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with ReLU non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return nn.functional.relu(self.linear(input))
    
class PosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            #assert fn_samples is not None
            fn_samples = sidelength
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = 4

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)
    
class ReLU_PE(nn.Module):
    def __init__(self, in_features,
                 hidden_features, hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        
        self.complex = False
        self.nonlin = ReLULayer
        self.out_features = out_features
            
        if pos_encode:
            self.positional_encoding = PosEncoding(in_features=in_features,
                                                   sidelength=sidelength,
                                                   fn_samples=fn_samples,
                                                   use_nyquist=use_nyquist)
            in_features = self.positional_encoding.out_dim
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            if self.complex:
                dtype = torch.cfloat
            else:
                dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
                        
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
            
        output = self.net(coords)
                    
        return output.view(-1,self.out_features)


############################################################################################################################
class GaussLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.scale = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return gaussian_core(self.linear(input), self.scale)
    
class GAUSS(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, scale=30.0):
        super().__init__()
        self.nonlin = GaussLayer
        
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, scale=scale))

        final_linear = nn.Linear(hidden_features, out_features)                
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output    

############################################################################################################################
