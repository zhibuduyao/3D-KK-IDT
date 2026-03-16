import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import tqdm
import beam_propagation_method as fw
import wave_propagation_method as wpm
import os
import tifffile
import system_parameter as sp

def create_image():
    print("正在创建样本...")
    rr=4
    z_coords = torch.arange(sp.nz*2//1, device=sp.device, dtype=torch.float32)
    x_coords = torch.arange(sp.nx, device=sp.device, dtype=torch.float32)
    y_coords = torch.arange(sp.ny, device=sp.device, dtype=torch.float32)
    z_grid, x_grid, y_grid = torch.meshgrid(z_coords, x_coords, y_coords, indexing='ij')
    dist_sq = (z_grid - sp.nz)**2 + (x_grid - sp.nx//2)**2 + (y_grid - sp.ny//2)**2
    n_sample = torch.full((sp.nz*2//1, sp.nx, sp.ny), sp.n_medium, device=sp.device)
    n_sample[dist_sq < rr**2] = 1.331
    dist_sq = (z_grid - sp.nz)**2 + (x_grid - sp.nx//4)**2 + (y_grid - sp.ny//4)**2
    n_sample[dist_sq < rr**2] = 1.331

    print("样本图像创建中...")
    save_dir = './data/test/intensity'
    os.makedirs(save_dir, exist_ok=True)
    N_ill=48
    theta_z_deg=50
    
    for i in tqdm.tqdm(range(N_ill)):
        for j in range(sp.nz):
            defocus_distance = (j-sp.nz//2)*sp.objz_pixel_size
            intensity=torch.zeros((sp.nx,sp.ny)).to(sp.device)
            
            model1=wpm.WavePropagationMethod(623,sp.NA_obj,1.33,1.33,i*360/N_ill,theta_z_deg,sp.objx_pixel_size,sp.objz_pixel_size/2,20,sp.device)
            model2=fw.BeamPropagationMethod(623,sp.NA_obj,1.33,1.33,i*360/N_ill,theta_z_deg,sp.objx_pixel_size,sp.objz_pixel_size/2,20,sp.device)
            intensity=model2.forward(n_sample,defocus_distance)
            
        
            # torch.save(intensity,f'./data/intensity/{i}.pt')
            intensity_np = intensity.cpu().numpy()
            tifffile.imwrite(f'{save_dir}/xdeg{i*360/N_ill:.1f}_zdeg{theta_z_deg}_z{defocus_distance}.tiff', intensity_np)




def draw_image(i):
    intensity=torch.load(f'./data/intensity/{i}.pt')
    if intensity.is_cuda:
        img_data = intensity.detach().cpu().numpy()
    else:
        img_data = intensity.numpy()
    plt.imshow(img_data, cmap='gray')
    plt.axis('off')
    plt.savefig(f'./data/show_{i}.png', dpi=300, bbox_inches='tight', pad_inches=0)

create_image()
# for i in range(16):
#     draw_image(i)
