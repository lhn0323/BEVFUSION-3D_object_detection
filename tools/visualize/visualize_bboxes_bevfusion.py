#!/usr/bin/env python3
# visualize_bevfusion_tracking_full.py
import argparse
import os
import math
from typing import List, Optional

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
from scipy.optimize import linear_sum_assignment

OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize BEVFusion 3D boxes projected on multi-camera original images with light tracking."
    )
    parser.add_argument("config", help="Path to BEVFusion config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.01, help="Score threshold for predictions")
    parser.add_argument("--step", type=int, default=-1, help="Number of steps to visualize (-1 for full dataloader)")
    parser.add_argument("--cam_order", type=int, nargs="*", default=None, help="Camera order indices")
    parser.add_argument("--fps", type=float, default=2.0, help="Output video FPS")
    parser.add_argument("--out", type=str, default="visualize_bevfusion_tracking.mp4", help="Output video path")
    parser.add_argument("--show-pred", action="store_true", help="Show predictions (with tracking). If not set, show GT boxes.")
    parser.add_argument("--max-vis", type=int, default=10, help="Only visualize first K frames (-1 for all). Default 10")
    parser.add_argument("--only-classes", nargs="*", default=None, help="Only visualize these class names (e.g. truck car). Default None (all)")
    return parser.parse_args()


# -------------------------
# Simple Tracker (center-distance + class gating)
# -------------------------
class Track:
    def __init__(self, det: np.ndarray, track_id: int):
        # det: ndarray shape (9,) [x,y,z,dx,dy,dz,yaw,score,label]
        self.det = det.copy()
        self.id = track_id
        self.missed = 0


class Tracker3D:
    """Light-weight tracker: center-distance matching + class gating.
    Does not change box coordinates — stable for visualization.
    """
    def __init__(self, max_missed=3, max_dist=4.0):
        self.max_missed = max_missed
        self.max_dist = max_dist
        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections: np.ndarray):
        """detections: np.ndarray (N,9) or empty (0,9)."""
        dets = np.asarray(detections)
        if dets.size == 0:
            # mark all missed
            for t in self.tracks:
                t.missed += 1
            self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
            return self.get_results()

        if len(self.tracks) == 0:
            for det in dets:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1
            return self.get_results()

        N = len(self.tracks)
        M = dets.shape[0]
        cost = np.full((N, M), fill_value=1e6, dtype=np.float32)
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(dets):
                try:
                    t_label = int(t.det[8])
                    d_label = int(d[8])
                except Exception:
                    t_label = d_label = None
                if (t_label is not None) and (d_label is not None) and (t_label != d_label):
                    # class mismatch -> heavy cost
                    continue
                cost[i, j] = np.linalg.norm(t.det[:2] - d[:2])

        row, col = linear_sum_assignment(cost)
        matched_tracks = set()
        matched_dets = set()
        for r, c in zip(row, col):
            if cost[r, c] <= self.max_dist:
                self.tracks[r].det = dets[c].copy()
                self.tracks[r].missed = 0
                matched_tracks.add(r)
                matched_dets.add(c)

        for i in range(len(self.tracks)):
            if i not in matched_tracks:
                self.tracks[i].missed += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        for j in range(M):
            if j not in matched_dets:
                self.tracks.append(Track(dets[j], self.next_id))
                self.next_id += 1

        return self.get_results()

    def get_results(self):
        # returns list of {"id": id, "box": ndarray(9,)}
        return [{"id": t.id, "box": t.det.copy()} for t in self.tracks]


# -------------------------
# Projection / Drawing
# -------------------------
def project_to_image(points: np.ndarray, lidar2cam: np.ndarray, cam2img: np.ndarray):
    """Project 3D points (N,3) in LiDAR coords to 2D image coordinates using
       lidar2cam (4x4) and cam2img (3x4 or 3x3 or 4x4).
       Returns (N,2), valid_mask (N,)
    """
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    pts_cam = (lidar2cam @ points_h.T).T  # (N,4)
    valid = pts_cam[:, 2] > 0

    cam2img = np.array(cam2img)
    # if cam2img is 4x4, use first 3 rows
    if cam2img.shape[0] == 4 and cam2img.shape[1] == 4:
        proj = (cam2img[:3, :] @ np.hstack((pts_cam[:, :3], np.ones((pts_cam.shape[0], 1)))).T).T
    else:
        # expect (3,4) or (3,3)
        if cam2img.shape[0] == 3 and cam2img.shape[1] == 4:
            proj = (cam2img @ np.hstack((pts_cam[:, :3], np.ones((pts_cam.shape[0], 1)))).T).T
        elif cam2img.shape[0] == 3 and cam2img.shape[1] == 3:
            proj = (cam2img @ pts_cam[:, :3].T).T
        else:
            # fallback: try first 3 rows
            proj = (cam2img[:3, :] @ np.hstack((pts_cam[:, :3], np.ones((pts_cam.shape[0], 1)))).T).T

    # normalize
    proj2 = proj[:, :2] / (proj[:, 2:3] + 1e-8)
    return proj2, valid


def draw_projected_3d_bboxes(ax, boxes_3d: np.ndarray, labels: np.ndarray, lidar2cam: np.ndarray, cam2img: np.ndarray, img_shape, classes, linewidth=2):
    """Draw 3D boxes (N,7) with labels (N,) onto ax of original image using ori_cam2img."""
    if boxes_3d is None or len(boxes_3d) == 0:
        return
    H, W = img_shape[:2]
    # convert to LiDARInstance3DBoxes to get corners
    # boxes_3d is (N,7) -> LiDARInstance3DBoxes expects tensor
    boxes_tensor = torch.from_numpy(np.array(boxes_3d)).float()
    lidar_boxes = LiDARInstance3DBoxes(boxes_tensor)  # CPU ok
    corners = lidar_boxes.corners.cpu().numpy()  # (N,8,3)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for i in range(corners.shape[0]):
        corner = corners[i]
        cls_idx = int(labels[i]) if labels is not None else 0
        cls_name = classes[cls_idx] if classes is not None and cls_idx < len(classes) else "unknown"
        color = np.array(OBJECT_PALETTE.get(cls_name, (255, 255, 255))) / 255.0

        proj2d, valid = project_to_image(corner, lidar2cam, cam2img)
        # require some corners in front
        if not valid.any():
            continue
        for (a, b) in edges:
            p1 = proj2d[a]
            p2 = proj2d[b]
            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue
            # optionally clip lines to image bounds
            x1, y1 = p1; x2, y2 = p2
            if (x1 < -10000) or (x2 < -10000):
                continue
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth)


def infer_grid(n_views: int):
    # prefer 3 columns for readability
    cols = 3 if n_views > 2 else n_views
    rows = int(math.ceil(n_views / max(1, cols)))
    return rows, cols


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    init_default_scope("mmdet3d")

    cfg = Config.fromfile(args.config)
    # ensure small batch for visualization
    cfg.test_dataloader.batch_size = 1
    cfg.val_dataloader.batch_size = 1

    classes = cfg.class_names

    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    # build model and load weights
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(get_device())
    model.eval()

    # prepare output and tracker
    os.makedirs(os.path.dirname(args.out) if args.out else "./", exist_ok=True)
    video_writer = None
    steps = args.step if args.step != -1 else (len(dataloader) if hasattr(dataloader, "__len__") else 10)
    max_steps = args.max_vis if args.max_vis > 0 else steps

    tracker = Tracker3D(max_missed=3, max_dist=4.0)

    # map class names to indices if user requested filter
    only_class_idxs: Optional[List[int]] = None
    if args.only_classes is not None and len(args.only_classes) > 0:
        only_class_idxs = []
        for cname in args.only_classes:
            if cname in classes:
                only_class_idxs.append(classes.index(cname))
            else:
                print(f"Warning: class name '{cname}' not in config class list.")
        if len(only_class_idxs) == 0:
            only_class_idxs = None

    data_iter = iter(dataloader)
    pbar = tqdm(range(max_steps), desc="frames")
    for frame_idx in pbar:
        try:
            data = next(data_iter)
        except StopIteration:
            break

        with torch.no_grad():
            outputs = model.test_step(data)

        sample = data["data_samples"][0]
        # metadata fields you printed: ori_cam2img, cam2img, lidar2cam, img_aug_matrix, img_path, pad_shape, batch_input_shape
        # We'll project to the **original** images using ori_cam2img and lidar2cam.
        img_paths = getattr(sample, "img_path", None)
        ori_cam2img = getattr(sample, "ori_cam2img", None)  # expected shape (n_cam, 4, 4) or (n_cam, 3, 4/3)
        lidar2cam_list = getattr(sample, "lidar2cam", None)

        if img_paths is None or ori_cam2img is None or lidar2cam_list is None:
            print("Missing camera metadata (img_path / ori_cam2img / lidar2cam). Skipping frame.")
            continue

        if args.show_pred:
            pred = outputs[0].pred_instances_3d
            bboxes3d = pred["bboxes_3d"].tensor.detach().cpu().numpy()  # shape (N,7+)
            labels = pred["labels_3d"].cpu().numpy() if "labels_3d" in pred else np.zeros((len(bboxes3d),), dtype=np.int32)
            scores = pred["scores_3d"].cpu().numpy() if "scores_3d" in pred else np.ones((len(bboxes3d),), dtype=np.float32)

            keep_mask = scores >= args.threshold
            if keep_mask.sum() == 0:
                dets = np.zeros((0, 9), dtype=np.float32)
                tracked = tracker.update(dets)
            else:
                bboxes3d = bboxes3d[keep_mask]
                labels = labels[keep_mask]
                scores = scores[keep_mask]
                # optionally filter classes
                if only_class_idxs is not None:
                    keep = [i for i, lab in enumerate(labels) if int(lab) in only_class_idxs]
                    if len(keep) == 0:
                        dets = np.zeros((0, 9), dtype=np.float32)
                        tracked = tracker.update(dets)
                    else:
                        bboxes3d = bboxes3d[keep]
                        labels = labels[keep]
                        scores = scores[keep]

                # assemble detections for tracker: [x,y,z,dx,dy,dz,yaw,score,label]
                # bboxes3d[:, :7] typically (x,y,z,dx,dy,dz,yaw)
                dets = np.concatenate([bboxes3d[:, :7], scores[:, None], labels[:, None]], axis=1)
                tracked = tracker.update(dets)
        else:
            # Show GT (no tracking)
            gt = sample.eval_ann_info
            if gt is None:
                print("No GT for this sample. Skipping.")
                continue
            bboxes3d = gt["gt_bboxes_3d"].tensor.cpu().numpy()
            labels = gt["gt_labels_3d"]
            # optionally filter classes
            if only_class_idxs is not None:
                keep = [i for i, lab in enumerate(labels) if int(lab) in only_class_idxs]
                if len(keep) == 0:
                    tracked = []
                else:
                    filtered_boxes = bboxes3d[keep]
                    filtered_labels = labels[keep]
                    tracked = [{"id": i + 1, "box": np.concatenate([filtered_boxes[i][:7], [1.0, filtered_labels[i]]])} for i in range(len(filtered_boxes))]
            else:
                tracked = [{"id": i + 1, "box": np.concatenate([bboxes3d[i][:7], [1.0, labels[i]]])} for i in range(len(bboxes3d))]

        # build visualization grid and draw per camera
        num_views = len(img_paths)
        cam_order = args.cam_order if args.cam_order is not None else list(range(num_views))
        rows, cols = infer_grid(len(cam_order))
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axs = np.array(axs).reshape(-1)

        for i, cam_id in enumerate(cam_order):
            if i >= len(axs):
                break
            ax = axs[i]
            img_path = img_paths[cam_id]
            img = cv2.imread(img_path)
            if img is None:
                ax.axis("off")
                ax.set_title(f"Cam {cam_id} (missing)")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # lidar2cam and ori_cam2img for this camera
            lidar2cam = np.array(lidar2cam_list[cam_id], dtype=np.float64)
            cam2img = np.array(ori_cam2img[cam_id], dtype=np.float64)

            # We have tracked list of {"id": id, "box": ndarray(9,)}
            # convert tracked list -> arrays for drawing
            if len(tracked) == 0:
                boxes_for_draw = np.zeros((0, 7))
                labels_for_draw = np.zeros((0,), dtype=np.int32)
            else:
                boxes_for_draw = np.array([t["box"][:7] for t in tracked])
                labels_for_draw = np.array([int(t["box"][8]) for t in tracked], dtype=np.int32)

            ax.imshow(img_rgb)
            draw_projected_3d_bboxes(ax, boxes_for_draw, labels_for_draw, lidar2cam, cam2img, img_rgb.shape, classes)
            ax.axis("off")
            ax.set_title(f"Cam {cam_id}")

        # hide unused axes
        for j in range(len(cam_order), len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        plt.close(fig)

        if video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
        video_writer.write(frame)

    if video_writer is not None:
        video_writer.release()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
