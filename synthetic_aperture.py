import torch
import tifffile
import system_parameter as sp

def synthetic_aperture(comphase, angle_index, z_index):
    # 初始化复散射势傅里叶谱
    O_syn_hat = torch.zeros((sp.nz, sp.ny, sp.nx), dtype=torch.complex64, device=sp.device)
    weight_syn = torch.zeros((sp.nz, sp.ny, sp.nx), device=sp.device) 

    u_m = torch.tensor(sp.n_medium / sp.wave_length).to(sp.device) 
    du =100*min(sp.dux, sp.duy, sp.duz) 

    for i in range(comphase.shape[0]):
        ## 1. 计算入射波的空间频率
        theta = angle_index[i][1] * torch.pi / 180
        phi = angle_index[i][0] * torch.pi / 180
        u_inx = -torch.tensor(torch.sin(theta) * torch.cos(phi) / sp.wave_length).to(sp.device)
        u_iny = -torch.tensor(torch.sin(theta) * torch.sin(phi) / sp.wave_length).to(sp.device)
        u_inz = torch.tensor(torch.sqrt(u_m**2 - u_inx**2 - u_iny**2 + 0j).real).to(sp.device)

        phi_s_3d = comphase[i]
        phi_s_hat = torch.fft.fftshift(torch.fft.fftn(phi_s_3d))
    
        ## 2. 计算支撑区条件
        U_prime_X = sp.UX + u_inx
        U_prime_Y = sp.UY + u_iny
        U_prime_Z = sp.UZ + u_inz

        on_ewald = torch.abs(torch.sqrt(U_prime_X**2 + U_prime_Y**2 + U_prime_Z**2) - u_m) < du
        in_pupil = (torch.sqrt(U_prime_X**2 + U_prime_Y**2) <= sp.NA_obj / sp.wave_length)
        Mask = (on_ewald & in_pupil)

        ## 3. 改进：使用浮点坐标进行线性分配，替代 round()
        # 计算浮点索引
        fx = sp.UX / sp.dux + sp.nx // 2
        fy = sp.UY / sp.duy + sp.ny // 2
        fz = sp.UZ / sp.duz + sp.nz // 2

        # 仅处理有效区域
        valid_mask = (fx >= 0) & (fx < sp.nx-1) & (fy >= 0) & (fy < sp.ny-1) & (fz >= 0) & (fz < sp.nz-1) & Mask
        
        if valid_mask.sum() == 0:
            continue

        # 提取有效点的浮点坐标和源值
        v_fx, v_fy, v_fz = fx[valid_mask], fy[valid_mask], fz[valid_mask]
        source_values = phi_s_hat[valid_mask] * (4j * torch.pi * sp.UZ[valid_mask])

        # 计算 8 个邻近整数坐标
        x0, y0, z0 = torch.floor(v_fx).long(), torch.floor(v_fy).long(), torch.floor(v_fz).long()
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        # 计算插值权重
        xd, yd, zd = v_fx - x0, v_fy - y0, v_fz - z0

        # 定义 8 个角的权重和位置 (Trilinear interpolation weights)
        weights = [
            (1-zd)*(1-yd)*(1-xd), (1-zd)*(1-yd)*xd, (1-zd)*yd*(1-xd), (1-zd)*yd*xd,
            zd*(1-yd)*(1-xd), zd*(1-yd)*xd, zd*yd*(1-xd), zd*yd*xd
        ]
        coords = [
            (z0, y0, x0), (z0, y0, x1), (z0, y1, x0), (z0, y1, x1),
            (z1, y0, x0), (z1, y0, x1), (z1, y1, x0), (z1, y1, x1)
        ]

        # 批量累加到网格中
        for w, (cz, cy, cx) in zip(weights, coords):
            O_syn_hat.index_put_((cz, cy, cx), source_values * w.to(O_syn_hat.dtype), accumulate=True)
            weight_syn.index_put_((cz, cy, cx), w.to(weight_syn.dtype), accumulate=True)

    # 4. 归一化并计算逆变换
    O_syn_hat = torch.where(weight_syn > 1e-6, O_syn_hat / weight_syn, 0.0j)
    O_syn = torch.fft.ifftn(torch.fft.ifftshift(O_syn_hat))
    
    return O_syn
