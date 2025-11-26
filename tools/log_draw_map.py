import re
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 日志文件路径
log_file = "work_dirs/training_results/lidar_cam/20251119_175053/20251119_175053.log"

class_names = ['car', 'truck','bus', 'bicycle', 'pedestrian']

# 保存每个类别的 mAP
categories_map = defaultdict(list)
epochs_list = []

with open(log_file, "r") as f:
    lines = f.readlines()

epoch_num = None

for i, line in enumerate(lines):
    # 匹配 Epoch 信息
    m_epoch = re.search(r"Epoch\(val\)\s*\[(\d+)\]", line)
    if m_epoch:
        epoch_num = int(m_epoch.group(1))

    # 匹配 NuScenes metric header
    if "NuScenes metric/result:" in line:
        header_line = lines[i+1]
        headers = [h.strip() for h in header_line.split("|") if h.strip()]
        cat_index = headers.index("class_name")
        map_index = headers.index("mAP")

        j = i + 2
        while j < len(lines):
            row = lines[j]
            if not row.strip().startswith("|"):
                break
            row_items = [x.strip() for x in row.split("|") if x.strip()]
            if len(row_items) < max(cat_index, map_index) + 1:
                j += 1
                continue
            cat = row_items[cat_index]
            if cat in class_names:
                mp = row_items[map_index]
                try:
                    mp_val = float(mp)
                except:
                    mp_val = 0.0
                categories_map[cat].append((epoch_num, mp_val))
            j += 1
        if epoch_num not in epochs_list:
            epochs_list.append(epoch_num)

# 计算每个 epoch 的平均 mAP（只考虑指定类别）
overall_map = []
for ep in epochs_list:
    ep_values = []
    for cat in class_names:
        vals = categories_map.get(cat, [])
        for e, mp in vals:
            if e == ep:
                ep_values.append(mp)
    overall_map.append((ep, np.mean(ep_values) if ep_values else 0.0))

# 画图
plt.figure(figsize=(12,7))

for cat in class_names:
    values = categories_map[cat]
    values.sort(key=lambda x: x[0])
    ep, mp_vals = zip(*values)
    plt.plot(ep, mp_vals, marker='o', label=f"{cat} mAP")
    # 在每个点上标注数值
    for x, y in zip(ep, mp_vals):
        plt.text(x, y + 0.5, f"{y:.1f}", ha='center', va='bottom', fontsize=9)

# 总体平均 mAP
ep_overall, mp_overall = zip(*overall_map)
plt.plot(ep_overall, mp_overall, marker='x', linestyle='--', color='black', linewidth=2, label="Overall mAP")
for x, y in zip(ep_overall, mp_overall):
    plt.text(x, y + 0.5, f"{y:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel("Epoch")
plt.ylabel("mAP (%)")
plt.title("BEVFusion NuScenes mAP over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 写入 TensorBoard
writer = SummaryWriter("work_dirs/tensorboard")
for cat in class_names:
    for ep, mp_val in categories_map[cat]:
        writer.add_scalar(f"val/mAP_{cat}", mp_val, ep)

for ep, mp_val in overall_map:
    writer.add_scalar("val/mAP_overall", mp_val, ep)

writer.close()
print("完成！TensorBoard 日志已生成在 work_dirs/tensorboard，指定类别及总体 mAP 曲线已绘制。")