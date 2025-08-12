#!/usr/bin/env python3
import sys
#sys.path.append(os.path.dirname(os.path.abspath(__file__)), "..")

import os
import glob
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cv2
import tf
from typing import List, Dict, Tuple, Any, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import scipy.sparse as sp

STATIC_LABELS   = {"room", "building"}

class SceneGraphMerger:
    def __init__(self, merged_dir, data_root, out_json, voxel_size=0.05, nn_radius=0.2):
        self.static_labels = {"room", } #"building"
        self.STATIC_NEI = {"building", "room", "window"}
        self.merged_dir = merged_dir
        self.data_root = data_root
        self.out_json = out_json
        self.voxel_size = voxel_size
        self.nn_radius = nn_radius
        self.cluster_eps = 1.5 * self.nn_radius     # DBSCAN 반경 (m) ~= 1~2*nn_radius
        self.cluster_min_samples = 15 #getattr(self, "cluster_min_samples", 20)
        self.cluster_max_points = 10000 #getattr(self, "cluster_max_points", 10000)
        self.vocab = None

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
        bboxes = self.nodes[node_id].get("bboxes", {})
        for _id, depth, pose in zip(dp["id"], dp["depth"], dp["pose"]):
            bbox = bboxes.get(_id)
            if bbox is None:
                continue
            pc = self.extract_pointcloud_from_bbox(depth, pose, bbox)
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
    
    def _cluster_pc(self, P: np.ndarray):
        P = np.asarray(P, np.float32).reshape(-1, 3)
        N = P.shape[0]
        if N == 0:
            return []
        
        if N > self.cluster_max_points:
            idx = np.random.choice(N, self.cluster_max_points, replace=False)
            P = P[idx]
            N = P.shape[0]

        min_samples = self.cluster_min_samples
        if N <= min_samples:
            return [P]

        eps = self.cluster_eps
        
        adj_min_samples = max(2, min_samples)  # 혹은 max(2, int(0.01*N))

        db = DBSCAN(eps=eps, min_samples=adj_min_samples).fit(P)
        labels = db.labels_
        clusters = [P[labels == k] for k in np.unique(labels) if k != -1]

        return clusters if clusters else [P]

    def _nnratio_oneway_aligned(self, A, B):
        if len(A)==0 or len(B)==0: return 0.0
        ca, cb = A.mean(axis=0), B.mean(axis=0)
        A0, B0 = A - ca, B - cb
        tree = cKDTree(B0)
        neighs = tree.query_ball_point(A0, r=self.nn_radius)
        hits = sum(1 for idxs in neighs if idxs)
        return hits / len(neighs)

    def nnratio(self, pc1, pc2):
        pc1 = np.asarray(pc1, np.float32).reshape(-1,3)
        pc2 = np.asarray(pc2, np.float32).reshape(-1,3)
        if pc1.size==0 or pc2.size==0: return 0.0

        cl1 = self._cluster_pc(pc1)
        cl2 = self._cluster_pc(pc2)
        best = 0.0
        for A in cl1:
            for B in cl2:
                r12 = self._nnratio_oneway_aligned(A,B)
                r21 = self._nnratio_oneway_aligned(B,A)
                best = max(best, r12, r21)
                # 혹은 mean? 아니면 median? 
        return float(best)
    
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
    
    def build_boc_tfidf(self, 
        *,
        min_df: int = 1,               # 몇 개 이상의 노드에서 등장하는 토큰만 사용
        use_confidence: bool = True,   # confidence를 TF 가중치로 사용할지
        sublinear_tf: bool = True,     # TF -> 1+log(TF)
        l2_normalize: bool = True,     # 최종 행벡터 L2 정규화
        max_features: Optional[int] = None  # 상위 DF/TF-idf 기준으로 자를 때 사용 가능
    ) :

        static_labels = ["room", "building", ] #"table"
        def _static_label(_id):
            label = self.nodes[_id].get("label")
            if label in static_labels:
                return label
            else :
                return _id
        
        def _add_token(bag: Dict[str, float], token: str, w: float):
            bag[token] = bag.get(token, 0.0) + float(w)

        nodes = set()
        node_bags: Dict[str, Dict[str, float]] = defaultdict(dict)

        for e in self.edges:
            s = e["subject"]; o = e["object"]; p = e["predicate"]
            c = float(e.get("confidence", 1.0))
            w = c if use_confidence else 1.0

            nodes.add(s); nodes.add(o)

            # merging subject as table, vase, ..
            tok_out = f"out:{p}:{_static_label(o)}"
            _add_token(node_bags[s], tok_out, w)

            # merging subject as table, vase, ..
            tok_in = f"in:{p}:{_static_label(s)}"
            _add_token(node_bags[o], tok_in, w)

        node_order = sorted(nodes)
        N = len(node_order)

        df_counter = Counter()
        for u in node_order:
            for t in node_bags[u].keys():
                df_counter[t] += 1
        
        tokens = [t for t, df in df_counter.items() if df >= min_df]
        
        if max_features is not None and len(tokens) > max_features:
            # -df_counter[t] to order tokens as descend order, t to order as alphabatical order 
            tokens = sorted(tokens, key=lambda t: (-df_counter[t], t))[:max_features]
        
        vocab = {t: i for i, t in enumerate(sorted(tokens))}
        rows, cols, data = [], [], []

        idf = np.zeros(len(vocab), dtype=np.float32)
        for t, j in vocab.items():
            df = df_counter[t]
            idf[j] = np.log((N + 1) / (df + 1)) + 1.0  # smoothed IDF

        for i, u in enumerate(node_order):
            bag = node_bags[u]
            if not bag:
                continue
            
            for t, tf in bag.items():
                j = vocab.get(t)
                if j is None:
                    continue
                val = tf
                if sublinear_tf:
                    val = 1.0 + np.log(max(val, 1e-12))
                rows.append(i); cols.append(j); data.append(val * idf[j])

        X = sp.csr_matrix((data, (rows, cols)), shape=(N, len(vocab)), dtype=np.float32)

        if l2_normalize and X.nnz > 0:
            norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
            nz = norms > 0
            X[nz] = sp.diags(1.0 / norms[nz]) @ X[nz]

        for i, node_id in enumerate(node_order):
            self.nodes[node_id]["relation"] = X[i,:].toarray().reshape(len(vocab), 1)
        
        return vocab

    def relation_cosine(self, u_vec, v_vec, vocab, remove_static=True):
        if u_vec is None or v_vec is None: 
            return 0.0
        sim_all = float(u_vec.T @ v_vec)

        if not remove_static: 
            return sim_all
        
        mask = np.ones((len(vocab), 1), dtype=np.float32)
        for tok, j in vocab.items():
            parts = tok.split(":")
            if len(parts) >= 3:
                nei = parts[-1]
                if any(nei.startswith(s) for s in self.STATIC_NEI):
                    mask[j, 0] = 0.0

        u = u_vec * mask
        v = v_vec * mask
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu == 0 or nv == 0: 
            return 0.0
        return float((u.T @ v) / (nu * nv))

    def node_sim(self, id1, id2):
        if not self._pair_valid(id1, id2):
            return 0.0
        pc_sim = self.nnratio(self.nodes[id1]["pc"], self.nodes[id2]["pc"])
        rel_vec1 = self.nodes[id1].get("relation")
        rel_vec2 = self.nodes[id2].get("relation")
        if rel_vec1 is None or rel_vec2 is None:
            relation_sim = 0.0
        else:
            relation_sim = self.relation_cosine(rel_vec1, rel_vec2, self.vocab)
        sim = 0.4 * pc_sim + 0.6 * relation_sim
        return sim

    def update_node_features(self):
        for node_id in list(self.nodes.keys()):
            self.extract_pc(node_id)
            # should add another method to update other features
        self.vocab = self.build_boc_tfidf()
        print("updated node features")

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

    def merge_pair(self, ui, uj, sim_score = 0.0, allow_static = True):
        if allow_static and (self._is_static(ui) or self._is_static(uj)):
            return

        self.nodes[ui]["sim"] = sim_score
        
        # pc_u = self.nodes[ui]["pc"]
        # pc_v = self.nodes[uj]["pc"]
        
        # merged_pc = np.vstack([pc_u, pc_v]) if pc_u.size and pc_v.size else (pc_u if pc_v.size == 0 else pc_v)
        # merged_pc = self.voxel_downsample(merged_pc, self.voxel_size)
        #self.nodes[ui]["pc"] = merged_pc

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

    def iterative_merge(self, threshold=0.5):
        sim = self.compute_all_sim()
        rooms_id = [id for id in self.nodes.keys() if self.nodes[id].get('label') == "room"]
        target = sorted(rooms_id)[0]
        while len(rooms_id) > 1 :
            if target not in rooms_id:
                if rooms_id:
                    target = sorted(rooms_id)[0]
                else:
                    print("unexpected room merging finished")
                    continue
            candidates = sorted([nid for nid in rooms_id if nid != target])
            other = candidates[0]
            self.merge_pair(target, other, 1.0, allow_static = False)
            rooms_id = [id for id in self.nodes.keys() if self.nodes[id].get('label') == "room"]
            print(f"room merged as {target} <- {other}")

        print("room merged finshed")

        while True:
            candidates = [(pair, score) for pair, score in sim.items() if score >= threshold]
            if not candidates: break
            (ui, uj), _ = max(candidates, key=lambda x: x[1])
            self.merge_pair(ui, uj, sim[(ui, uj)])
            self.update_node_features()
            print(f"{(ui, uj)} merged, similarity {sim[(ui, uj)]}")
            sim = self.compute_all_sim()
    
    def _json_default(self, o):
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

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
            pc = node.get("pc")
            if isinstance(pc, np.ndarray) and pc.shape[0] > 0:
                idx = np.random.choice(pc.shape[0], 10, replace=False)
                pc_selected = pc[idx].tolist()

            new_nodes.append({
                "id": new_id,
                "label": node["label"],
                "pc" : pc_selected
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
            json.dump(final, f, indent=2, default = self._json_default)

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
