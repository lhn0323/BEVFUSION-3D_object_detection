import argparse
import os
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmengine.device import get_device
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from tqdm import tqdm

# 定义对象调色板，包含 truck
OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71), # truck 的颜色
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

# --- 辅助函数保持不变 ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize BEVFusion predictions by projecting 3D bboxes onto multi-view camera images and saving a video."
    )
    parser.add_argument("config", help="Path to BEVFusion config file")
    parser.add_argument("--checkpoint", type=str, default="work_dirs/bevfusion_lidar_camera/20251115_085333/epoch_6.pth")
    parser.add_argument("--threshold", type=float, default=0.01, help="Score threshold for predictions")
    parser.add_argument("--step", type=int, default=40, help="Number of steps to visualize (-1 for full)")
    parser.add_argument("--cam_order", type=int, nargs="*", default=None, help="Custom camera order, e.g., 0 1 2 3")
    parser.add_argument("--fps", type=float, default=2.0, help="FPS of output video")
    parser.add_argument("--out", type=str, default="work_dirs/visualization/camera/visualization_camera_tracked.mp4", help="Output video path")
    parser.add_argument("--show-pred", action="store_true", help="Show predictions instead of ground truth")
    parser.add_argument("--track-mode", action="store_true", help="Enable basic tracking smoothing for flickering reduction.")
    return parser.parse_args()


def project_to_image(points, lidar2cam, cam2img):
    """Project 3D points to image plane, supporting 3x3 or 4x4 cam2img."""
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = np.dot(lidar2cam, points_hom.T).T
    valid_mask = points_cam[:, 2] > 0

    if cam2img.shape == (3, 3):
        proj = np.dot(cam2img, points_cam[:, :3].T).T
        proj = proj / proj[:, 2:3]
        return proj[:, :2], valid_mask
    elif cam2img.shape == (4, 4):
        points_cam_h = np.hstack((points_cam[:, :3], np.ones((points_cam.shape[0], 1))))
        proj_h = np.dot(cam2img, points_cam_h.T).T
        proj = proj_h[:, :3] / proj_h[:, 2:3]
        return proj[:, :2], valid_mask
    else:
        raise ValueError(f"Unsupported cam2img shape: {cam2img.shape}")


def draw_projected_3d_bboxes(ax, bboxes, labels, lidar2cam, cam2img, img_shape, classes, linewidth=1):
    """
    Projects 3D bounding boxes to 2D and draws them on the given axis with class-specific colors.
    """
    H, W = img_shape[:2]

    def clip_line(pt1, pt2, W, H):
        """Clip a line segment to the image bounds."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Both points are inside
        if (0 <= x1 < W and 0 <= y1 < H) and (0 <= x2 < W and 0 <= y2 < H):
            return [pt1, pt2]

        # If both are out of bounds, skip
        if (x1 < 0 and x2 < 0) or (x1 >= W and x2 >= W) or (y1 < 0 and y2 < 0) or (y1 >= H and y2 >= H):
            return None

        # Simple clipping
        pt1_clipped = [np.clip(x1, 0, W - 1), np.clip(y1, 0, H - 1)]
        pt2_clipped = [np.clip(x2, 0, W - 1), np.clip(y2, 0, H - 1)]
        return [pt1_clipped, pt2_clipped]

    if bboxes is None or len(bboxes) == 0:
        return

    # only take the first 7 dims for corners
    lidar_boxes = LiDARInstance3DBoxes(bboxes[:, :7])
    corners_3d = lidar_boxes.corners.cpu().numpy()  # (N, 8, 3)

    # Edges of a 3D bbox
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)   # verticals
    ]

    for idx, corners in enumerate(corners_3d):
        name = classes[labels[idx]]
        color = np.array(OBJECT_PALETTE[name]) / 255
        
        projected, valid_mask = project_to_image(corners, lidar2cam, cam2img)
        if not valid_mask.all():
            continue
        for j, k in lines:
            pt1, pt2 = projected[j], projected[k]
            clipped = clip_line(pt1, pt2, W, H)
            if clipped:
                ax.plot([clipped[0][0], clipped[1][0]],
                        [clipped[0][1], clipped[1][1]],
                        color=color, linewidth=linewidth)


def infer_grid(n_views: int):
    # Prefer 2x3 for 6, 2x2 for 4; otherwise near-square grid
    if n_views == 6:
        return 2, 3
    if n_views == 4:
        return 2, 2
    rows = int(math.floor(math.sqrt(n_views)))
    cols = int(math.ceil(n_views / rows))
    return rows, cols

# --- 新增的跟踪历史和匹配逻辑 ---

class SimpleTrackHistory:
    """一个简单的类用于存储前一帧的检测结果，以便进行帧间平滑。"""
    def __init__(self, max_frames_missed=3):
        # {track_id: {'bbox': tensor, 'label': int, 'missed': int}}
        self.active_tracks = {}
        self.max_frames_missed = max_frames_missed
        self.next_available_id = 0

    def update_tracks(self, current_bboxes, current_labels, current_scores):
        """
        接收当前帧的检测结果，并返回一个包含平滑后的 bbox/label 的列表。
        """
        if current_bboxes.shape[0] == 0:
            # 如果当前帧没有检测到任何东西，则只衰减历史记录
            self._decay_missed_tracks()
            return [], []

        # 1. 如果模型提供了 track_ids，则直接使用它
        if 'track_ids' in current_scores.keys():
            track_ids = current_scores['track_ids'].cpu().numpy()
            
            # 清空旧的未匹配计数
            for track_id in self.active_tracks:
                self.active_tracks[track_id]['matched'] = False
                self.active_tracks[track_id]['missed'] += 1

            # 更新匹配的轨道，并添加新的轨道
            for bbox, label, track_id in zip(current_bboxes, current_labels, track_ids):
                if track_id not in self.active_tracks:
                    # 这是一个新的轨道
                    self.active_tracks[track_id] = {
                        'bbox': bbox,
                        'label': label,
                        'missed': 0,
                        'matched': True
                    }
                else:
                    # 更新现有轨道
                    self.active_tracks[track_id]['bbox'] = bbox
                    self.active_tracks[track_id]['label'] = label
                    self.active_tracks[track_id]['missed'] = 0
                    self.active_tracks[track_id]['matched'] = True

            # 2. 衰减/清理未匹配的轨道
            self._decay_missed_tracks()
            
            # 3. 返回所有激活的轨道
            smoothed_bboxes = [v['bbox'] for v in self.active_tracks.values()]
            smoothed_labels = [v['label'] for v in self.active_tracks.values()]
            return smoothed_bboxes, smoothed_labels
        
        # 否则，我们跳过简单的距离匹配（这需要更复杂的逻辑，例如 Hungarian 匹配和距离计算），
        # 仅在检测中启用 'track-mode' 时，保持前一帧未被当前帧覆盖的物体。
        else:
            # 在没有 ID 的情况下，我们不做复杂的匹配，只保留当前检测结果。
            # 要减少闪烁，我们需要一个 ID 匹配机制。
            # 为了简单起见，我们直接返回当前帧的检测结果，跳过此处的平滑。
            # 专业的跟踪平滑需要一个独立的跟踪器（例如 MOT/SORT/AB3DMOT）。
            
            # 此时 'track-mode' 仅为占位符，因为没有 ID 无法进行平滑。
            return current_bboxes.tolist(), current_labels.tolist()


    def _decay_missed_tracks(self):
        """删除连续多帧未匹配的轨道，保留未超过阈值的轨道。"""
        tracks_to_remove = []
        for track_id, track_data in self.active_tracks.items():
            if track_data['missed'] > self.max_frames_missed:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]

# --- 主函数 ---

# ... (imports, OBJECT_PALETTE, parse_args, project_to_image, draw_projected_3d_bboxes, infer_grid, SimpleTrackHistory 类保持不变) ...

def main():
    args = parse_args()
    init_default_scope("mmdet3d")

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.test_dataloader.batch_size = 1

    classes = cfg.class_names

    # dataloader
    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    # model
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(get_device())
    model.eval()

    os.makedirs(os.path.join(cfg.get("work_dir", "./work_dirs/default"), "visualization"), exist_ok=True)
    video_path = args.out or os.path.join(cfg.get("work_dir", "./work_dirs/default"),
                                          "visualization",
                                          "visualization_bevfusion_tracked.mp4")

    # 初始化跟踪器
    track_history = SimpleTrackHistory(max_frames_missed=5) # 允许物体连续失踪 5 帧

    video_writer = None
    
    # =======================================================
    # 核心修改：将可视化的步数限制为 10 帧
    # 即使 args.step 是 -1 (或大于 10)，也只取 10
    total_dataloader_steps = len(dataloader)
    
    # 优先使用 args.step, 但不能超过 dataloader 总长度。
    # 如果 args.step 是 -1 或大于 10，则将其设置为 10 (除非总长度更小)。
    if args.step != -1 and args.step < total_dataloader_steps:
        steps = args.step
    else:
        steps = total_dataloader_steps
        
    # 强制限制最大帧数为 10
    steps = min(steps, 300)
    # =======================================================

    pbar = tqdm(range(steps), desc=f"Visualizing {steps} Frames")
    data_iter = iter(dataloader)
    for _ in pbar:
        data = next(data_iter)  # each iteration yields one batch
        
        # --- 1. 获取检测结果或真值 ---
        if args.show_pred:
            with torch.no_grad():
                outputs = model.test_step(data)
            
            pred = outputs[0].pred_instances_3d
            bboxes = pred["bboxes_3d"].tensor.detach().cpu()
            labels = pred["labels_3d"].detach().cpu() if "labels_3d" in pred else None
            scores = pred["scores_3d"].detach().cpu() if "scores_3d" in pred else None
            
            if scores is not None and args.threshold is not None:
                # 过滤低分检测结果
                keep = scores >= args.threshold
                bboxes = bboxes[keep]
                if labels is not None:
                    labels = labels[keep]
                scores_filtered = {k: v[keep] for k, v in pred.items() if k.endswith('3d') and k != 'bboxes_3d'}
            else:
                 scores_filtered = {}


            # --- 2. 启用跟踪平滑 ---
            if args.track_mode:
                # 使用跟踪器更新和获取平滑后的 bboxes/labels
                smoothed_bboxes, smoothed_labels = track_history.update_tracks(bboxes, labels, scores_filtered)
                
                # 更新 bboxes 和 labels 为平滑后的结果
                if len(smoothed_bboxes) > 0:
                    bboxes = torch.stack(smoothed_bboxes)
                    labels = torch.tensor(smoothed_labels)
                else:
                    bboxes = torch.empty((0, 9))
                    labels = torch.empty(0, dtype=torch.long)
            # --- 否则，直接使用当前帧的检测结果 (默认行为) ---

        else:
            # 可视化地面真值 (无需跟踪平滑)
            gt_data = data['data_samples'][0].eval_ann_info
            bboxes = gt_data['gt_bboxes_3d'].tensor.detach().cpu()
            labels = gt_data['gt_labels_3d']

        # --- 3. 渲染可视化图像 ---
        
        # sample meta
        sample = data["data_samples"][0]
        # img_path: list[str]; lidar2cam/cam2img: list[np.ndarray]
        img_paths = getattr(sample, "img_path", None)
        lidar2cam_list = getattr(sample, "lidar2cam", None)
        cam2img_list = getattr(sample, "cam2img", None)

        if img_paths is None or lidar2cam_list is None or cam2img_list is None:
            print("Missing camera metadata (img_path/lidar2cam/cam2img). Skipping frame.")
            continue

        num_views = len(img_paths)
        cam_order = args.cam_order if args.cam_order is not None else list(range(num_views))
        rows, cols = infer_grid(len(cam_order))

        # build figure
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axs = np.array(axs).reshape(-1)
        for i, cam_id in enumerate(cam_order):
            if i >= len(axs):
                break
            img_path = img_paths[cam_id]
            img = cv2.imread(img_path)
            if img is None:
                axs[i].axis("off")
                axs[i].set_title(f"Camera {cam_id} (missing)")
                continue
            # BGR -> RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            lidar2cam = np.array(lidar2cam_list[cam_id], dtype=np.float64)
            cam2img = np.array(cam2img_list[cam_id], dtype=np.float64)

            ax = axs[i]
            ax.imshow(img_rgb)
            draw_projected_3d_bboxes(ax, bboxes, labels, lidar2cam, cam2img, img_rgb.shape, classes)
            ax.axis("off")
            ax.set_title(f"Camera {cam_id}")

        # hide unused axes
        for j in range(len(cam_order), len(axs)):
            axs[j].axis("off")

        plt.tight_layout()

        # figure -> frame
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        plt.close(fig)

        # init writer after first frame (match exact size)
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = frame.shape[:2]
            video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))

        video_writer.write(frame)
        
        # 达到限制步数后退出循环
        if _ + 1 >= steps:
            break

    if video_writer is not None:
        video_writer.release()
    print(f"Saved video (first {steps} frames) to: {video_path}")


if __name__ == "__main__":
    main()