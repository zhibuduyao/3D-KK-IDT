项目旨在通过获取三维光强堆栈，实现折射率三维重建。

## 1. 环境配置

matplotlib                3.10.6 
numpy                     2.2.6 
torch                     2.9.1+cu126
torchvision               0.24.1+cu126
tqdm                      4.67.1

## 2. 项目结构

1.main.py: 主程序，用于读取四维光强数组（第一维是角度，第二维是高度，第三第四维是xy平面），并调用重建函数。读取的时候会记录光强数组对应的照明角度与高度索引，用于后续重建时使用。

2.beam_propagation_method.py: 基于BPM方法实现多重散射物理模拟输出光强图像，调用方法：
```python
import beam_propagation_method as bpm
model=bpm.BeamPropagationMethod(wavelength,NA_obs,n_oil,n_medium,theta_x_deg, theta_z_deg,pixel_size_x,pixel_size_z,amplitude,device)
intensity=model.forward(n_sample)
```
其中，wavelength为波长，NA_obs为物镜数值孔径，n_oil为油膜折射率，n_medium为介质折射率，theta_x_deg和theta_z_deg为入射角，pixel_size_x和pixel_size_z为横向与轴向样品处像素大小，amplitude为振幅，device为计算设备，n_sample为样本三维折射率分布。

3.wave_propagation_method.py: 基于WPM方法实现多重散射物理模拟输出光强图像，调用方法与BPM类似

4.system_prameter.py: 定义系统参数，包括波长、物镜数值孔径、油膜折射率、介质折射率、横向与轴向样品处像素大小、计算设备、三维光强数组的第n层对应的z轴高度。所有长度单位均为纳米。

5.cal_complex_phase.py: 利用光强对数谱得到实部，然后利用k-k关系计算虚部，输入为LED入射角度及在该角度下的三维光强堆栈（.pt文件），输出为三维复相位分布。这里的三维光强堆栈第一维是z向，第二维与第三维是x与y向。

调用方法：
```python
import cal_complex_phase as ccp
comphase=ccp.cal_complex_phase(angle_x_deg,angle_z_deg,intensity,z_index)
```
其中，angle_x_deg与angle_z_deg为LED入射角度，intensity为三维光强堆栈（第一维是z轴，第二维第三维是zy轴，如果想要第三层的光强，直接intensity[2]即可。第i层的高度为z_index[i-1]）。

6.synthetic_aperture.py: 合成孔径,输入为四维多角度LED复相位分布以及角度索引，输出为三维折射率分布并保存。

7.data文件夹下只能有文件夹，不能有文件，文件夹名为样品或项目名称。在项目文件夹下，有intensity文件夹：用于存放二维光强图像（图像命名：xdeg{a}_zdeg{b}_z{c}.tiff，表示这张图像是在LED在三维坐标轴中与x轴成a度，z轴成b度拍摄的，样品在距离焦平面c nm高度处），RI文件夹：用于存放重建之后的三维折射率（.pt文件）