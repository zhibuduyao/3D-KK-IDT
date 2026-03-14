import cal_complex_phase as ccp
import system_parameter as sp
import torch
import synthetic_aperture as sa
import data_loader as dl
import tifffile

img_list,angle_index,z_index=dl.load_tiff('./data/test',sp.device) #读取文件
img_list.to(sp.device)
angle_index.to(sp.device)
z_index.to(sp.device)
# print(z_index)

#提取每个角度的复相位
comphase=torch.zeros_like(img_list,dtype=torch.complex64)
for i in range(img_list.shape[0]):
    comphase[i]=ccp.cal_complex_phase(angle_index[i][0],angle_index[i][1],img_list[i],z_index)

#计算合成孔径
O=sa.synthetic_aperture(comphase,angle_index,z_index)*1e-2
n=torch.sqrt(O/(2*torch.pi/sp.wave_length)**2+sp.n_medium**2).real #根据散射势计算折射率

tifffile.imwrite('./data/test/n.tiff',n.cpu().numpy())
tifffile.imwrite('./data/test/comphase.tiff',comphase[0].imag.cpu().numpy())
tifffile.imwrite('./data/test/img.tiff',img_list[0].cpu().numpy())
print("合成孔径计算完成，正在保存结果...")
print(n.real.max().item(),n.real.min().item())


#图像优化

print("Done!")
