import re
import tifffile
import torch
from pathlib import Path
from collections import defaultdict

def load_tiff(project_path, device='cuda'):
    """加载TIFF文件为PyTorch四维张量"""
    
    folder = Path(project_path) / "intensity"
    pattern = re.compile(r'xdeg([-]?\d*\.?\d+)_zdeg([-]?\d*\.?\d+)_z([-]?\d*\.?\d+)\.tiff')
    
    # 按(x_angle, z_angle)分组
    groups = defaultdict(dict)  # {(x,z): {height: file}}
    
    for f in folder.glob("*.tiff"):
        match = pattern.match(f.name)
        if match:
            x, z, h = map(float, match.groups())
            groups[(x, z)][h] = f
    
    if not groups:
        return None, None, None
    
    # 获取排序后的角度组合和高度
    angle_combos = sorted(groups.keys())  # [(x1,z1), (x2,z2), ...]
    all_heights = sorted(set(h for heights in groups.values() for h in heights))
    
    # 读取第一张图片获取尺寸
    first_file = next(iter(groups.values()))[next(iter(all_heights))]
    img = tifffile.imread(first_file)
    H, W = img.shape
    
    # 创建四维张量
    img_stack = torch.zeros((len(angle_combos), len(all_heights), H, W), 
                           dtype=torch.float32, device=device)
    
    # 创建角度列表
    angle_list = torch.tensor(angle_combos, dtype=torch.float32, device=device)  # (n, 2)
    
    # 填充张量
    for i, (x, z) in enumerate(angle_combos):
        for j, h in enumerate(all_heights):
            if h in groups[(x, z)]:
                img_np = tifffile.imread(groups[(x, z)][h])
                img_stack[i, j] = torch.from_numpy(img_np).to(img_stack.dtype)
    
    # 创建高度列表
    height_list = torch.tensor(all_heights, dtype=torch.float32, device=device)
    
    return img_stack, angle_list, height_list