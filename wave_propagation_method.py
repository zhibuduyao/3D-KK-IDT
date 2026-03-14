import torch
import torch.fft as fft
import numpy as np
import math
import matplotlib.pyplot as plt


class WavePropagationMethod(torch.nn.Module):
    """
    实现光束传播法作为前向模型
    """
    def __init__(self,wavelength,NA_obs,n_oil,n_medium,theta_x_deg, theta_z_deg,pixel_size_x,pixel_size_z,amplitude,device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        """
        wavelength: 波长
        NA_obs: 物镜数值孔径
        n_oil: 物镜浸润油折射率
        n_medium: 介质折射率
        theta_x_deg: 入射角x方向(单位: 度)
        theta_z_deg: 入射角z方向(单位: 度)
        pixel_size_x: x方向像素大小
        pixel_size_z: z方向像素大小
        device: 计算设备
        """
        super(WavePropagationMethod, self).__init__()
        self.wavelength = wavelength
        self.NA_obs = NA_obs
        self.n_oil = n_oil
        self.n_medium = n_medium
        self.pixel_size_x = pixel_size_x
        self.pixel_size_z = pixel_size_z
        self.amplitude=amplitude
        self.device = device

        self.theta_x = torch.tensor(theta_x_deg/180.0*math.pi,device=device)
        self.theta_z = torch.tensor(theta_z_deg/180.0*math.pi,device=device)

        self.k0 = 2 * math.pi / wavelength
        self.k_medium = self.k0 * n_medium

        self.kx_incident = -self.k0 * torch.sin(self.theta_z) * torch.cos(self.theta_x)
        self.ky_incident = -self.k0 * torch.sin(self.theta_z) * torch.sin(self.theta_x)

        self.pad_mag=2 # 向四周padding的长度/原本样本的边长

        self.input_field = None
    
    def wpm_one_layer(self,input_field, dz, n_layer):
        field = input_field.clone()
        field_f =fft.fft2(field)
        kx = 2 * torch.pi * fft.fftfreq(field.shape[-2], self.pixel_size_x, device=self.device)
        ky = 2 * torch.pi * fft.fftfreq(field.shape[-1], self.pixel_size_x, device=self.device)
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        field_output=torch.zeros_like(field,device=self.device,dtype=field.dtype)
        unique_refractive_indices = torch.unique(n_layer).to(self.device)
        for n_value in unique_refractive_indices:

            kz_m = torch.sqrt( (self.k0 * n_value)**2 - KX**2 - KY**2 + 0j) # 确保为复数
            propagation_phase = torch.exp(1j * kz_m * dz)
            
            # 3. 严格的论文公式：先统一传播，再在实空间用指示函数选取
            propagated_field = fft.ifft2(field_f * propagation_phase)
            
            # 生成当前折射率区域的指示函数
            I_m = (n_layer == n_value)
            # 论文公式：用指示函数加权叠加
            field_output += I_m * propagated_field
        return field_output

            
    
    def create_tilted_plane_wave(self,orig_height, orig_width,padded_height, padded_width):
        total_height = orig_height + 2 * padded_height  
        total_width = orig_width + 2 * padded_width    
        y = (torch.arange(total_height, device=self.device) - total_height//2) * self.pixel_size_x
        x = (torch.arange(total_width, device=self.device) - total_width//2) * self.pixel_size_x
        Y, X = torch.meshgrid(y, x, indexing='ij')
        
        # 生成倾斜平面波
        phase = self.kx_incident * X + self.ky_incident * Y
        field = self.amplitude * torch.exp(1j * phase)
        
        # 创建渐变掩模（只在填充区域有渐变，原始样本区域保持1.0）
        mask = torch.ones(total_height, total_width, device=self.device)
        

        # 创建距离图
        y_coord, x_coord = torch.meshgrid(
            torch.arange(total_height, device=self.device), 
            torch.arange(total_width, device=self.device), 
            indexing='ij'
        )


        # 取最小距离（正值表示在中心区域外）
        dist_to_edge = torch.min(
        torch.min(y_coord, total_height - 1 - y_coord),
        torch.min(x_coord, total_width - 1 - x_coord))

        # 创建渐变掩模
        grad_mask = dist_to_edge < min(orig_height, orig_width)

        if grad_mask.any():
            max_dist = min(orig_height, orig_width)
            normalized_dist = dist_to_edge[grad_mask].float() / max_dist
            cos_weights = (1 - torch.cos(normalized_dist * math.pi))/2
            mask[grad_mask] = cos_weights

        # 应用掩模到光场
        field = field * mask

        return field
    
    def get_camera_primary(self,input_field,nz,defocus_distance):

        """
        得到焦平面的光强图像
        """
        field=input_field.clone()
        if isinstance(field, np.ndarray):
            field = torch.from_numpy(field).to(self.device)
        else:
            field = field.to(self.device)

        nx, ny = field.shape
        # 生成频率坐标
        fx = fft.fftfreq(nx, self.pixel_size_x, device=self.device)
        fy = fft.fftfreq(ny, self.pixel_size_x, device=self.device)
        FY, FX = torch.meshgrid(fy, fx, indexing='ij')
        kx=2*math.pi*FX
        ky=2*math.pi*FY

        kz_squared = self.k_medium**2 - kx**2 - ky**2
        # 计算频率半径
        freq_radius = torch.sqrt(FX**2 + FY**2)

        # 相机截止频率
        cutoff_frequency = self.NA_obs / self.wavelength

        # 创建相机低通滤波器
        camera_filter = torch.zeros_like(freq_radius, device=self.device)
        camera_filter[freq_radius <= cutoff_frequency] = 1.0
    
        # 应用滤波 - 使用PyTorch FFT
        field_f = fft.fft2(field)

        field_filtered_f = field_f * camera_filter
        # 数字重聚焦
        
        distance = -self.pixel_size_z* nz/2.0-defocus_distance
        # distance = 0
        
            
        H = torch.zeros_like(kz_squared, dtype=torch.complex64, device=self.device)

        # 传播波（kz_squared >= 0）
        mask_propagating = kz_squared >= 0
        kz_propagating = torch.sqrt(kz_squared[mask_propagating] + 0j)  # 确保为复数
        H[mask_propagating] = torch.exp(1j * kz_propagating * distance)
        # 倏逝波（kz_squared < 0）
        mask_evanescent = kz_squared < 0
        if distance > 0:  # 正向传播，倏逝波衰减
            kz_evanescent = torch.sqrt(-kz_squared[mask_evanescent])
            H[mask_evanescent] = torch.exp(-kz_evanescent[mask_evanescent] * distance)
        else:  # 反向传播，滤除倏逝波（无法恢复）
            H[mask_evanescent] = 0.0
        
        
        # 应用传递函数
        field_refocused_f = field_filtered_f * H
        field = fft.ifft2(field_refocused_f)
        
        return field.abs()**2



    def forward(self,n_sample,defocus_distance):
        """
        返回出射场的光强
        """
        if isinstance(n_sample, np.ndarray):
            n_sample = torch.from_numpy(n_sample).to(self.device)
        else:
            n_sample = n_sample.to(self.device)
            
        depth, orig_height, orig_width = n_sample.shape
        dz = self.pixel_size_z
        pad_height = int(round(orig_height * self.pad_mag))
        pad_width = int(round(orig_width * self.pad_mag))
        if self.input_field is None:
            self.input_field=self.create_tilted_plane_wave(orig_height, orig_width, pad_height, pad_width)
        
        n_sample_padded = torch.nn.functional.pad(
            n_sample, 
            (pad_width, pad_width, pad_height, pad_height), 
            mode='constant', 
            value=self.n_medium
        )
        # 初始化光场
        field = self.input_field.clone()
        # 沿z轴逐层传播 - 实现论文中的递归BPM
        for z in range(depth):
            # 获取当前层的折射率
            n_layer = n_sample_padded[z, :, :]  # [batch_size, height, width]
            # WPM步骤: 逐层传播
            field = self.wpm_one_layer(field, dz, n_layer)
            
        intensity=self.get_camera_primary(field,n_sample.shape[0],defocus_distance)
        return intensity[pad_height:pad_height+orig_height, pad_width:pad_width+orig_width]
    
