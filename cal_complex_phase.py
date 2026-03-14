import torch
import torch.fft as fft
import math
import system_parameter as sp  # 第四部分参数（待设定）！！！

def cal_complex_phase(angle_x_deg, angle_z_deg, intensity, z_index):
    """"
    一阶近似Rytov和KK滤波
    
        angle_x_deg: 入射角 x 方向 (单位: 度)
        angle_z_deg: 入射角 z 方向 (单位: 度)
        intensity: 三维光强堆栈, 形状为 (Nz, Nx, Ny)
        z_index: z轴信息
    
    """
    device = sp.device
    if not isinstance(intensity, torch.Tensor):
        intensity = torch.tensor(intensity, device=device)
    if not isinstance(z_index, torch.Tensor):
        z_index = torch.tensor(z_index, device=device)

    # 三维xyz坐标
    Nz, Nx, Ny = intensity.shape

    k0 = 2 * math.pi / sp.wave_length
    
    theta_x = angle_x_deg / 180.0 * math.pi
    theta_z = angle_z_deg / 180.0 * math.pi

    kx_in, ky_in =-torch.tensor(torch.sin(theta_z)*torch.cos(theta_x)/sp.wave_length).to(sp.device), -torch.tensor(torch.sin(theta_z)*torch.sin(theta_x)/sp.wave_length).to(sp.device)
    kz_in=torch.tensor(torch.sqrt((k0*sp.n_medium)**2-kx_in**2-ky_in**2+0j).real).to(sp.device)
    #坐标定义
    
    KX = 2 * math.pi * sp.UX
    KY = 2 * math.pi * sp.UY
    KZ = 2 * math.pi * sp.UZ

    # Rytov 对数模型
    I_mean = intensity.mean()
    I_norm = intensity / I_mean
    ln_I = torch.log(I_norm + 1e-6)

    ln_I_hat = fft.fftshift(fft.fftn(ln_I))

    # 3Dkk滤波
    q_dot_uin = KX * kx_in + KY * ky_in + KZ * kz_in
    
    # 半空间提取
    H_half = (q_dot_uin < 0).float()
    H_half[torch.abs(q_dot_uin) < 1e-6] = 0.5 

    # 相位提取
    Phi_hat = ln_I_hat * H_half

    # ifft
    complex_phase = fft.ifftn(fft.ifftshift(Phi_hat))

    return complex_phase