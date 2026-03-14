import cal_complex_phase as ccp
import system_parameter as sp
import torch
import tifffile

def synthetic_aperture(comphase,angle_index,z_index):
    """
    comphase:复相位数组，维度为[照明角度，z轴扫描数量，图像高度，图像宽度]
    angle_index:照明角度的索引，二维数组，维度为[照明角度数量,2],分别为与x轴和z轴的夹角，单位为度
    z_index:z轴扫描高度的索引
    return:合成孔径后给出折射率，三维，维度为[z轴扫描数量，图像高度，图像宽度]
    """

    O_syn_hat = torch.zeros((sp.nz, sp.ny, sp.nx), dtype=torch.complex64, device=sp.device)#初始化复散射势傅里叶谱
    weight_syn = torch.zeros((sp.nz, sp.ny, sp.nx), device=sp.device)  # 用于记录每个频率点被填充的次数，用于后续平均

    u_m=torch.tensor(sp.n_medium/sp.wave_length).to(sp.device) #介质波数
    du=min(sp.dux,sp.duy,sp.duz) #空间频率网格间距量级

    for i in range(comphase.shape[0]):
        ## 计算入射波的空间频率
        theta = angle_index[i][1] * torch.pi / 180
        phi = angle_index[i][0] * torch.pi / 180
        u_inx, u_iny =-torch.tensor(torch.sin(theta)*torch.cos(phi)/sp.wave_length).to(sp.device), -torch.tensor(torch.sin(theta)*torch.sin(phi)/sp.wave_length).to(sp.device)
        u_inz=torch.tensor(torch.sqrt(u_m**2-u_inx**2-u_iny**2+0j).real).to(sp.device)
        if torch.abs(u_inz)<1e-6:
            print(f"警告⚠️：入射波接近临界角，可能导致数值不稳定，入射角度为theta={angle_index[i][1]}度, phi={angle_index[i][0]}度")

        phi_s_3d = comphase[i]
        phi_s_hat = torch.fft.fftn(phi_s_3d)
        phi_s_hat = torch.fft.fftshift(phi_s_hat) # 将零频率分量移到频谱中心
    
        

        ## 计算u'=u+u_in,然后判断是否满足支撑区条件
        U_prime_X=sp.UX+u_inx
        U_prime_Y=sp.UY+u_iny
        U_prime_Z=sp.UZ+u_inz

        #支撑区条件1：位于ewald球面附近
        on_ewald = torch.abs(torch.sqrt(U_prime_X**2 + U_prime_Y**2 + U_prime_Z**2) - u_m) < du

        # 支撑区条件2: 横向频率在瞳函数内
        in_pupil = (torch.sqrt(U_prime_X**2 + U_prime_Y**2) <= sp.NA_obj / sp.wave_length)

        Mask = (on_ewald & in_pupil).float()
        # print(f"照明角度theta={angle_index[i][1]}度, phi={angle_index[i][0]}度, 满足支撑区条件的频率点数量: {(Mask>0.5).sum().item()}")

        ## 映射复相位到复散射势
        #计算v=u-u_in
        VX = sp.UX
        VY = sp.UY
        VZ = sp.UZ

        #将频率转换为索引
        ix=torch.round(VX/sp.dux + sp.nx//2).long() 
        iy=torch.round(VY/sp.duy + sp.ny//2).long()
        iz=torch.round(VZ/sp.duz + sp.nz//2).long()
        # print(ix.max(), ix.min(), iy.max(), iy.min(), iz.max(), iz.min())
        valid_mask = (ix >= 0) & (ix < sp.nx) & (iy >= 0) & (iy < sp.ny) & (iz >= 0) & (iz < sp.nz) & (Mask > 0.5)
        if valid_mask.sum() == 0:
            print(f"警告⚠️：没有有效频率点满足支撑区条件，入射角度为theta={angle_index[i][1]}度, phi={angle_index[i][0]}度")
            continue

        source_values = phi_s_hat[valid_mask]*(4j*torch.pi*sp.UZ[valid_mask]) #添加因子项，后续可能需要考虑数据稳定性
        target_z_indices = iz[valid_mask]
        target_y_indices = iy[valid_mask]
        target_x_indices = ix[valid_mask]

        O_syn_hat.index_put_((target_z_indices,  target_y_indices,target_x_indices), source_values, accumulate=True)
        weight_syn.index_put_((target_z_indices, target_y_indices, target_x_indices), torch.ones_like(source_values, device=sp.device, dtype=weight_syn.dtype), accumulate=True)
        if i==0:
            tifffile.imwrite('./data/test/weight.tiff', weight_syn.real.cpu().numpy())
        

    O_syn_hat = torch.where(weight_syn > 1e-6, O_syn_hat / weight_syn, 0.0j)
    O_syn = torch.fft.ifftn(torch.fft.ifftshift(O_syn_hat))
    
    
    return O_syn
