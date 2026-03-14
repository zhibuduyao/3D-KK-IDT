import torch


NA_obj=0.8
wave_length = 623
n_medium = 1.33
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
cmos_pixel_size=6.5e3
magnification=40
objx_pixel_size=cmos_pixel_size/magnification
objy_pixel_size=cmos_pixel_size/magnification
objz_pixel_size=200

nx=512
ny=512
nz=99

### 创建一个空间频率网格，避免后面重复创建
ux = torch.fft.fftshift(torch.fft.fftfreq(nx, d=objx_pixel_size)).to(device)
uy = torch.fft.fftshift(torch.fft.fftfreq(ny, d=objy_pixel_size)).to(device)
uz = torch.fft.fftshift(torch.fft.fftfreq(nz, d=objz_pixel_size)).to(device)
UZ, UY, UX = torch.meshgrid(uz, uy, ux, indexing='ij')

dux = UX[0, 0, 1] - UX[0, 0, 0]
duy = UY[0, 1, 0] - UY[0, 0, 0]
duz = UZ[1, 0, 0] - UZ[0, 0, 0]
# print(f"空间频率网格间距：dux={dux:.4e}, duy={duy:.4e}, duz={duz:.4e}")

