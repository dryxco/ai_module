#!/usr/bin/env python3
import sys
#sys.path.append(os.path.dirname(os.path.abspath(__file__)), "..")

import os
import glob
import json
import math
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
import tf
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree

STATIC_LABELS   = {"room", "building"}

class SceneGraphMerger:
    def __init__(self, merged_dir, data_root, out_json, voxel_size=0.05, nn_radius=0.2):
        self.static_labels = {"room", "building"}
        self.merged_dir = merged_dir
        self.data_root = data_root
        self.out_json = out_json
        self.voxel_size = voxel_size
        self.nn_radius = nn_radius

        self.nodes = {}
        self.edges = []
        self.idp_pair = {}

    def load_merged_graphs(self):
        graphs = []
        for fp in sorted(glob.glob(os.path.join(self.merged_dir, "merged_sg_*.json"))):
            with open(fp, 'r') as f:
                graphs.append(json.load(f))
        return graphs

    def build_global_sg(self, graphs):
        for gi, g in enumerate(graphs):
            for n in g["nodes"]:
                uid = f"{n['id']}_{gi}"
                self.nodes[uid] = {
                    "id": uid,
                    "orig_id": n["id"],
                    "label": n["label"],
                    "bbox": n["bbox"],
                    "bboxes" : n["bboxes"],
                    "pc": None,
                    "img_stat" : None,
                }
            for e in g["edges"]:
                sid = f"{e['subject']}_{gi}"
                oid = f"{e['object']}_{gi}"
                self.edges.append({
                    "subject": sid,
                    "object": oid,
                    "predicate": e["predicate"],
                    "confidence": e.get("confidence", 1.0)
                })

    def align_depth_pose(self, node_idx):
        img_dirs = sorted(glob.glob(os.path.join(self.data_root, str(node_idx), "image", "*.png")))
        depth_dirs = sorted(glob.glob(os.path.join(self.data_root, str(node_idx), "depth", "*.png")))
        pose_paths = sorted(glob.glob(os.path.join(self.data_root, str(node_idx), "pose", "*.json")))
        if not depth_dirs or not pose_paths:
            return None
        if len(depth_dirs) != len(pose_paths):
            print(f"[Warning] Depth/pose count mismatch for node {node_idx}")
        idp = {"id":[], "image":[], "depth": [], "pose": []}
        for i_path, d_path, p_path in zip(img_dirs, depth_dirs, pose_paths):
            img = cv2.imread(i_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            with open(p_path, 'r') as f:
                pose = json.load(f)
            
            frame_id = Path(i_path).stem
            idp["id"].append(str(frame_id))
            idp["image"].append(img)
            idp["depth"].append(depth)
            idp["pose"].append(pose)
        self.idp_pair[node_idx] = idp

    def _is_static(self, node_id):
        return self.nodes[node_id].get("label") in self.static_labels
    
    def _pair_valid(self, id1, id2):
        if self._is_static(id1) or self._is_static(id2):
            return False
        
        if self.nodes[id1]["label"] != self.nodes[id2]["label"]:
            return False
        
        return True

    def extract_pointcloud_from_bbox(self, depth, pose, bbox, camera_offset_z=0.0):
        image_width, image_height = depth.shape[1], depth.shape[0]

        x0 = int(np.floor(bbox[0])); y0 = int(np.floor(bbox[1]))
        x1 = int(np.ceil (bbox[2])); y1 = int(np.ceil (bbox[3]))
        
        x0 = max(0, min(x0, image_width-1))
        x1 = max(0, min(x1, image_width))
        y0 = max(0, min(y0, image_height-1))
        y1 = max(0, min(y1, image_height))
        
        if x1 <= x0 or y1 <= y0:
            rospy.logwarn("Empty/invalid bbox after clipping")
            return np.empty((0,3), dtype=np.float32)
        
        patch = depth[y0:y1, x0:x1]
        h, w  = patch.shape
        ys = np.arange(y0, y0 + h)
        xs = np.arange(x0, x0 + w)
        u_coord, v_coord = np.meshgrid(xs, ys)

        d = patch.reshape(-1).astype(np.float32)
        u = u_coord.flatten()
        v = v_coord.flatten()

        mask = np.isfinite(d) & (d > 0)
        if not np.any(mask):
            return np.empty((0,3), dtype=np.float32)
        
        u, v, d = u[mask], v[mask], d[mask]
        d = d / 1000.0  # mm to meters

        theta = math.pi - 2 * math.pi * u / image_width
        phi = math.pi/2 - math.pi * v / image_height

        X = d * np.cos(phi) * np.cos(theta)
        Y = d * np.cos(phi) * np.sin(theta)
        Z = d * np.sin(phi)

        pts_cam = np.vstack([X, Y, Z, np.ones_like(X)])
        pos = pose["position"]; ori = pose["orientation"]
        T = (tf.transformations.translation_matrix([pos["x"], pos["y"], pos["z"] + camera_offset_z]) @
            tf.transformations.quaternion_matrix([ori["x"], ori["y"], ori["z"], ori["w"]]))
        pts_map = T @ pts_cam

        return pts_map[:3, :].T.astype(np.float32)

    def extract_pc(self, node_id):
        orig, idx = node_id.rsplit('_', 1)
        #idx = node_id[-1]
        
        idx = int(idx)
        dp = self.idp_pair.get(idx)
        if dp is None:
            self.nodes[node_id]["pc"] = np.empty((0, 3), dtype=np.float32)
            return
        pcs = []
        for _id, depth, pose in zip(dp["id"], dp["depth"], dp["pose"]):
            pc = self.extract_pointcloud_from_bbox(depth, pose, self.nodes[node_id]["bboxes"][_id])
            if pc.size:
                pcs.append(pc)
        if not pcs:
            self.nodes[node_id]["pc"] = np.empty((0, 3), dtype=np.float32)
            return
        all_pc = np.vstack(pcs)
        # Outlier removal
        mu, sig = all_pc.mean(0), all_pc.std(0)
        inliers = all_pc[np.all(np.abs(all_pc - mu) <= 2 * sig, axis=1)]
        # Voxel downsample
        keys = np.floor(inliers / self.voxel_size).astype(np.int32)
        _, idxs = np.unique(keys, axis=0, return_index=True)
        self.nodes[node_id]["pc"] = inliers[idxs]
    
    def nnratio_oneway(self, A, B):
        if A.size == 0 or B.size == 0:
            return 0.0
        A = np.asarray(A, dtype=np.float32).reshape(-1, 3)
        B = np.asarray(B, dtype=np.float32).reshape(-1, 3)
        tree = cKDTree(B)
        neighs = tree.query_ball_point(A, r=self.nn_radius)
        hits = sum(1 for idxs in neighs if idxs)
        return hits / len(neighs)

    def nnratio(self, pc1, pc2):
        r12 = self.nnratio_oneway(pc1, pc2)
        r21 = self.nnratio_oneway(pc2, pc1)
        return np.mean([r12, r21])
    
    def voxel_downsample(self, pc: np.ndarray, voxel_size: float = None) -> np.ndarray:
        """(N,3) 포인트클라우드를 voxel 격자(크기 m)로 다운샘플링해서
        각 voxel의 중심(centroid)을 대표점으로 반환."""
        if pc is None or pc.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        pc = np.asarray(pc, dtype=np.float32)
        mask = np.isfinite(pc).all(axis=1)
        pc = pc[mask]
        if pc.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        keys = np.floor(pc / voxel_size).astype(np.int64)  # (N,3)

        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        centroids = np.zeros((len(uniq), 3), dtype=np.float32)
        np.add.at(centroids, inv, pc)           # voxel별 좌표 합
        counts = np.bincount(inv)               # voxel별 점 개수
        centroids /= counts[:, None]            # 평균 = centroid

        return centroids
        
    def node_sim(self, id1, id2):
        if not self._pair_valid(id1, id2):
            return 0.0
        return self.nnratio(self.nodes[id1]["pc"], self.nodes[id2]["pc"])

    def update_node_features(self):
        for node_id in list(self.nodes.keys()):
            self.extract_pc(node_id)
            # should add another method to update other features

    def compute_all_sim(self):
        sim = {}
        ids = list(self.nodes.keys())
        for i, id_i in enumerate(ids):
            for id_j in ids[i+1:]:
                if not self._pair_valid(id_i, id_j):
                    continue
                sim[(id_i, id_j)] = self.node_sim(id_i, id_j)
                sim[(id_j, id_i)] = sim[(id_i, id_j)]
        return sim

    def merge_pair(self, ui, uj, sim_score = 0.0):
        if self._is_static(ui) or self._is_static(uj):
            return

        self.nodes[ui]["sim"] = sim_score
        
        pc_u = self.nodes[ui]["pc"]
        pc_v = self.nodes[uj]["pc"]
        
        merged_pc = np.vstack([pc_u, pc_v]) if pc_u.size and pc_v.size else (pc_u if pc_v.size == 0 else pc_v)
        merged_pc = self.voxel_downsample(merged_pc, self.voxel_size)

        self.nodes[ui]["pc"] = merged_pc
        # update edges
        new_edges = []
        for e in self.edges:
            s, o = e["subject"], e["object"]
            if s == uj: s = ui
            if o == uj: o = ui
            if s != o:
                new_edges.append({"subject": s, "object": o,
                                  "predicate": e["predicate"],
                                  "confidence": e["confidence"]})
        self.edges = new_edges
        # remove node
        del self.nodes[uj]

    def iterative_merge(self, threshold=0.9):
        sim = self.compute_all_sim()
        while True:
            candidates = [(pair, score) for pair, score in sim.items() if score >= threshold]
            if not candidates: break
            (ui, uj), _ = max(candidates, key=lambda x: x[1])
            self.merge_pair(ui, uj, sim[(ui, uj)])
            self.update_node_features()
            print(f"{(ui, uj)} merged, it similarity {sim[(ui, uj)]}")
            sim = self.compute_all_sim()

    def save_graph(self):
        # re assign node idx for all node id (flower1_0 => flower1)
        label_counter = Counter(n["label"] for n in self.nodes.values())

        next_idx = {label: 0 for label in label_counter}
        old2new = {}

        new_nodes = []
        for old_id, node in self.nodes.items():
            label = node["label"]
            idx   = next_idx[label]
            new_id = f"{label}{idx}"
            next_idx[label] += 1
            old2new[old_id] = new_id

            new_nodes.append({
                "id": new_id,
                "label": node["label"],
                #"bbox": node["bbox"]
            })
        
        new_edges = []
        for e in self.edges:
            s_new = old2new.get(e["subject"], e["subject"])
            o_new = old2new.get(e["object"],  e["object"])
            new_edges.append({
                "subject":   s_new,
                "object":    o_new,
                "predicate": e["predicate"],
                "confidence": e.get("confidence", 1.0)
            })

        final = {"nodes": new_nodes, "edges": new_edges}
        
        out_dir = os.path.dirname(self.out_json)
        os.makedirs(out_dir, exist_ok=True)
        with open(self.out_json, 'w') as f:
            json.dump(final, f, indent=2)

if __name__ == "__main__":
    merged_dir = "sgg/merged_sg"
    data_root = "ai_module/data"
    out_json = "sgg/final_sg.json"

    merger = SceneGraphMerger(merged_dir, data_root, out_json)
    graphs = merger.load_merged_graphs()
    merger.build_global_sg(graphs)

    # align image & depth & pose
    for idx in os.listdir(merged_dir):
        if idx.isdigit():
            merger.align_depth_pose(idx)

    merger.update_node_features()
    merger.iterative_merge(threshold=0.05)  # 필요 시 호출
    merger.save_graph()

    print(f"Scene graph saved to {out_json}")





# def merge_depth_for_node(node_idx, depth_root, ext=".png"):
#     depth_dir = os.path.join(depth_root, str(node_idx), "depth")
#     paths = sorted(glob.glob(os.path.join(depth_dir, f"*{ext}")))
#     if not paths:
#         return None

#     sum_d = None
#     cnt_d = None
#     for p in paths:
#         d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
#         valid = (d > 0).astype(np.float32)
#         if sum_d is None:
#             sum_d = d * valid
#             cnt_d = valid
#         else:
#             sum_d += d * valid
#             cnt_d += valid

#     # 평균 depth 계산. cnt_d는 누적한 횟수, sum_d는 누적 값에 해당
#     avg = np.zeros_like(sum_d)
#     mask = cnt_d > 0
#     avg[mask] = sum_d[mask] / cnt_d[mask]
#     return avg

    # def extract_statistics_patch(self, node_id,
    #                          hist_bins: int = 8) -> np.ndarray:
        
    #     orig, idx = node_id.rsplit('_', 1)
    #     idx = int(idx)
    #     dp = self.idp_pair.get(idx)

    #     if dp is None:
    #         self.nodes[node_id]["img_stat"] = np.zeros(1 + 3*hist_bins + 2, dtype=np.float32)
    #         return
        
    #     bbox = self.nodes[node_id]["bbox"]

    #     feat_list = []
    #     for img in dp["image"]:
    #         xmin, ymin, xmax, ymax = map(int, bbox)
    #         patch = img[ymin:ymax, xmin:xmax]

    #         if patch.size == 0:
    #             # [entropy, 3*hist_bins, meanV, meanS]
    #             return np.zeros(1 + 3*hist_bins + 2, dtype=np.float32)

    #         gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    #         hist_gray, _ = np.histogram(gray, bins=hist_bins, range=(0, 256))
    #         p = hist_gray.astype(np.float32)
    #         p_sum = p.sum()
    #         if p_sum > 0:
    #             p = p / p_sum
    #             p_nonzero = p[p > 0]
    #             entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
    #         else:
    #             entropy = 0.0

    #         hist_features = []
    #         for c in range(3):  # B, G, R
    #             channel = patch[:, :, c]
    #             hist_c, _ = np.histogram(channel, bins=hist_bins, range=(0, 256))
    #             # normalize
    #             if hist_c.sum() > 0:
    #                 hist_c = hist_c.astype(np.float32) / hist_c.sum()
    #             hist_features.append(hist_c)
    #         hist_features = np.concatenate(hist_features, axis=0)  # shape (3*hist_bins,)

    #         hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    #         # OpenCV HSV: H[0-179], S[0-255], V[0-255]
    #         S = hsv[:, :, 1] / 255.0
    #         V = hsv[:, :, 2] / 255.0
    #         mean_s = float(np.mean(S))
    #         mean_v = float(np.mean(V))

    #         feat = np.hstack([
    #             np.array([entropy], dtype=np.float32),
    #             hist_features.astype(np.float32),
    #             np.array([mean_v, mean_s], dtype=np.float32)
    #         ])

    #         feat_list.append(feat)

    #     img_stat = np.mean(np.stack(feat_list, axis=0), axis=0).astype(np.float32)
    #     self.nodes[node_id]["img_stat"] = img_stat