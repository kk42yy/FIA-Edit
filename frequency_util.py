import math
import torch
import torch.fft as fft


shape = (24,1,1024,64) # C, B, H, W
d_s = d_t = 0.9
T, H, W = shape[-3], shape[-2], shape[-1]
LPS = torch.zeros(shape)
for t in range(T):
    for h in range(H):
        for w in range(W):
            d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
            LPS[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)


def freq_com_SD3_v1(high, low, alpha=0.2, beta=None, eta=None, x_t_high=True, ds=0.3, dt=0.3):
    """ Frequency manipulation for latent space. """
    if alpha < 1e-5:
        return torch.randn_like(high).to(high.device)

    try:
        B, C, H, W = high.shape
        Three = False
    except:
        high = high[None, ...]
        low = low[None, ...]
        B, C, H, W = high.shape
        Three = True
    dtype = high.dtype
    high = high.view(C,B,H,W).to(torch.float32)
    low = low.view(C,B,H,W).to(torch.float32)
    f_shape = high.shape # 1, 4, 64, 64
    # LPF = get_freq_filter(f_shape, high.device, 'gaussian', n=4, d_s=ds, d_t=dt) # d_s, d_t
    LPF = LPS.to(high.device)
    f_dtype = high.dtype

    # High FFT
    HPF = 1 - LPF
    high_freq = fft.fftn(high, dim=(-3, -2, -1))
    high_freq = fft.fftshift(high_freq, dim=(-3, -2, -1))
    high_freq_high = high_freq * HPF
    
    # Low FFT
    low_freq = fft.fftn(low, dim=(-3, -2, -1))
    low_freq = fft.fftshift(low_freq, dim=(-3, -2, -1))
    low_freq_low = low_freq * LPF

    x_freq_sum = high_freq_high + alpha * low_freq_low #+ torch.randn_like(low_freq_low).to(low_freq_low.dtype)
    _x_freq_sum = fft.ifftshift(x_freq_sum, dim=(-3, -2, -1))
    x_sum = fft.ifftn(_x_freq_sum, dim=(-3, -2, -1)).real
    
    x_sum = x_sum.to(f_dtype)
    x_sum = x_sum.view(B,C,H,W)
    
    if Three:
        x_sum = x_sum[0]
    
    if beta is None:
        beta = 1 - alpha
        
    if eta is None:
        eta = alpha
        
    # Pos = torch.poisson(x_sum.abs())
    return (eta*x_sum + beta*torch.randn_like(x_sum).to(x_sum.device)).to(dtype)

def freq_com_SD3(high, low, alpha=0.2, beta=0., mode=2):
    """ Frequency manipulation for latent space. """
       
    try:
        B, C, H, W = high.shape
        Three = False
    except:
        high = high[None, ...]
        low = low[None, ...]
        B, C, H, W = high.shape
        Three = True
        
    dtype = high.dtype
    high = high.view(C,B,H,W).to(torch.float32)
    low = low.view(C,B,H,W).to(torch.float32)
    f_shape = high.shape # 1, 4, 64, 64
    LPF = LPS.to(high.device)
    f_dtype = high.dtype

    # High FFT
    HPF = 1 - LPF
    high_freq = fft.fftn(high, dim=(-3, -2, -1))
    high_freq = fft.fftshift(high_freq, dim=(-3, -2, -1))
    # Low FFT
    low_freq = fft.fftn(low, dim=(-3, -2, -1))
    low_freq = fft.fftshift(low_freq, dim=(-3, -2, -1))
    
        
    high_freq_high = high_freq * HPF
    high_freq_low = high_freq * LPF
    low_freq_low = low_freq * LPF
    low_freq_high = low_freq * HPF
    
    if mode == 1:
        x_freq_sum = high_freq_high + alpha * low_freq_low
        _x_freq_sum = fft.ifftshift(x_freq_sum, dim=(-3, -2, -1))
        x_sum = fft.ifftn(_x_freq_sum, dim=(-3, -2, -1)).real
        
        x_freq_sum2 = high_freq_low + alpha * low_freq_high
        _x_freq_sum2 = fft.ifftshift(x_freq_sum2, dim=(-3, -2, -1))
        x_sum2 = fft.ifftn(_x_freq_sum2, dim=(-3, -2, -1)).real
    
        x_sum = 0.5*(x_sum + x_sum2).to(f_dtype)
    else:
        x_freq_sum = high_freq_high + alpha * low_freq_low
        x_freq_sum2 = high_freq_low + alpha * low_freq_high
        x_freq_sum_total = 0.8 * x_freq_sum + 0.2 * x_freq_sum2
        
        _x_freq_sum_total = fft.ifftshift(x_freq_sum_total, dim=(-3, -2, -1))
        x_sum_total = fft.ifftn(_x_freq_sum_total, dim=(-3, -2, -1)).real
        
        x_sum = x_sum_total.to(f_dtype)
        
    x_sum = x_sum.view(B,C,H,W)
    
    if Three:
        x_sum = x_sum[0]

    return (x_sum + beta*torch.randn_like(x_sum)).to(dtype)