#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, sys, shutil
import rospy, rospkg
from std_msgs.msg import String, Int32MultiArray
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import glob

import hashlib, struct
from collections import defaultdict

import torch
from pathlib import Path
from PIL import Image as PILImage
import numpy as np

# alias 에러 방지
np.int   = int
np.float = float

PKG_ROOT = Path(rospkg.RosPack().get_path('reltr_scene_graph'))
# RelTR 루트를 파이썬 경로에 등록 (RelTR_SGG가 내부에서 RelTR.models.* import 함)
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
# sgg 모듈 추가
sys.path.insert(0, str(PKG_ROOT / 'sgg'))

from RelTR_SGG import build_model, load_checkpoint, infer_one_image
from merge import merge_folder  
from entire_merge import SceneGraphMerger

class RelTRSGGNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_topic   = rospy.get_param("~image_topic", "/camera/image")
        self.use_compressed= rospy.get_param("~use_compressed", False)
        self.weights       = rospy.get_param("~weights")
        self.device        = rospy.get_param("~device", "cuda")
        self.conf_th       = float(rospy.get_param("~confidence_thresh", 0.2))
        self.topk          = int(rospy.get_param("~topk_triplets", 10))
        self.save_dir      = os.path.expanduser(rospy.get_param("~save_dir", "~/.ros/scene_graph_results"))
        os.makedirs(self.save_dir, exist_ok=True)

        self.prev_edge_hash = None
        self.new_data = False

        pkg_root = Path(rospkg.RosPack().get_path('reltr_scene_graph'))
        data_dir = pkg_root.parents[2] / 'data'
        sgg_route = pkg_root.parents[0] / 'reltr_scene_graph/data'
        self.data_root = rospy.get_param("~data_root", str(data_dir))

        self.node_count = len(os.listdir(self.data_root))
        self.sgg_route = rospy.get_param("~sgg_route", str(sgg_route))

        # 모델 로드 (RelTR_SGG 방식)
        self.model = build_model()
        self.model = load_checkpoint(self.model, ckpt_path=self.weights)
        # use model as default setting if cuda is unavailable
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.if_process = False
        rospy.Subscriber('/exp_mode', String, self.mode_callback, queue_size=1)
        
        # to check if robot is in the new map
        rospy.Subscriber("/edge_list", Int32MultiArray, self.list_callback)

        if self.use_compressed:
            if self.new_data:
                rospy.Subscriber(self.image_topic, CompressedImage, self._on_image_compressed, queue_size=1, buff_size=2**24)
        else:
            rospy.Subscriber(self.image_topic, Image,           self._on_image_raw,        queue_size=1, buff_size=2**24)
        rospy.loginfo(f"[reltr_sgg] Subscribed to {self.image_topic}")
        
        self.pub_json = rospy.Publisher("scene_graph/json", String, queue_size=10)
        self.fin_pub = rospy.Publisher("/reltr_mode", String, queue_size = 1)
    
    def mode_callback(self, msg):
        if msg.data != "fin":
            self.if_process = False
        else :
            if not self.if_process :
                self.generate_all_scene_graphs()
                self.merge_all_graph()

                msg = String()
                msg.data = "fin"
                self.fin_pub.publish(msg)
                
                self.if_process = True
            if self.new_data:
                self.if_process = False
    
    def _on_image_raw(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._process_frame(frame, msg.header.stamp.to_nsec())
    
    def hash_list(self, msg):
        packed = struct.pack(f'{len(msg.data)}i', *msg.data)
        return hashlib.md5(packed).hexdigest()
    
    def list_callback(self, msg):
        current_hash = self.hash_list(msg)
        if current_hash == self.prev_edge_hash:
            self.new_data = False
            return
        rospy.loginfo("New edge_list received. load and process agian")
        self.new_data = True
        self.prev_edge_hash = current_hash

    def _on_image_compressed(self, msg: CompressedImage):
        frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        self._process_frame(frame, msg.header.stamp.to_nsec())

    @torch.no_grad()
    def _process_frame(self, frame_bgr, stamp_nsec: int):
        try:
            # OpenCV BGR -> PIL RGB
            img_rgb = frame_bgr[..., ::-1]
            pil_img = PILImage.fromarray(img_rgb)

            dev = next(self.model.parameters()).device
            trips = infer_one_image(self.model, pil_img, topk=self.topk, conf_th=self.conf_th, device=dev)

            # trips = infer_one_image(self.model, pil_img, topk=self.topk, conf_th=self.conf_th)

            out = {"stamp_nsec": int(stamp_nsec), "num": len(trips), "triplets": trips}
            out_path = os.path.join(self.save_dir, f"{stamp_nsec}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            self.pub_json.publish(String(data=json.dumps(out, ensure_ascii=False)))
            rospy.loginfo_throttle(2.0, f"[reltr_sgg] {len(trips)} triplets -> {out_path}")
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[reltr_sgg] frame failed: {type(e).__name__}: {e}")
    
    @torch.no_grad()
    def generate_all_scene_graphs(self):
        # erase previous merged scene graphs    
        for target_dir in [
            os.path.join(self.sgg_route, "merged_sg"),
            os.path.join(self.sgg_route, "sg_per_node")
        ]:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
                os.makedirs(target_dir, exist_ok=True)
        
        # to generate sg for every image in 1 node
        for idx in range(self.node_count):
            img_dirs = sorted(glob.glob(os.path.join(self.data_root, str(idx), "image", "*.png")))

            for _img_dir in img_dirs:
                try:
                    #img_file = cv2.imread(_img_dir, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    img_bgr = cv2.imread(_img_dir, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        raise RuntimeError(f"Failed to load image {_img_dir}")
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_rgb = img_rgb.astype(np.uint8)
                    pil_img = PILImage.fromarray(img_rgb)

                    dev    = next(self.model.parameters()).device
                    trips  = infer_one_image(self.model, pil_img,
                                            topk=self.topk,
                                            conf_th=self.conf_th,
                                            device=dev)
                    
                    stamp = Path(_img_dir).stem
                    out = {"stamp": stamp, "num": len(trips), "triplets": trips}
                    out_path = os.path.join(self.sgg_route, "sg_per_node", str(idx), f"sg_{stamp}.json")
                    out_dir = os.path.dirname(out_path)
                    os.makedirs(out_dir, exist_ok=True)
                    with open(out_path, "w") as f:
                        json.dump(trips, f, indent=2)
                except Exception as e:
                    print(f"{e}, error occured in generating sg in {_img_dir}")
                
                print(f"finished to generate sg in {idx} node of {_img_dir}")

        # to generate merged sg for each node
        for idx in range(self.node_count):
            json_dir = os.path.join(self.sgg_route, "sg_per_node", str(idx))
            merged_sg_json = os.path.join(self.sgg_route, "merged_sg", f"merged_sg_{idx}.json")
            merged_png  = os.path.join(self.sgg_route, "merged_sg", f"merged_sg_{idx}.png")

            merge_folder(json_dir=json_dir, out_json=merged_sg_json, out_png = merged_png)
        
        print("finished to generate merged sg for each node")
    
    @torch.no_grad()
    def merge_all_graph(self):
        merged_dir = os.path.join(self.sgg_route, "merged_sg")
        data_root = self.data_root
        out_json = os.path.join(self.sgg_route, "all_merged_sg", "all_merged_sg.json")

        all_merger = SceneGraphMerger(merged_dir, data_root, out_json)

        graphs = all_merger.load_merged_graphs()
        all_merger.build_global_sg(graphs)

        # align image & depth & pose
        for idx in os.listdir(data_root):
            try:
                idx = int(idx)
                all_merger.align_depth_pose(idx)
                print(f"{idx} depth, img, pose aligned")
            except Exception as e:
                print(f"{e}, error occurred in align data")
        
        all_merger.update_node_features()
        all_merger.iterative_merge(threshold=0.35)
        all_merger.save_graph()

        print(f"Scene graph saved to {out_json}")
    
def main():
    rospy.init_node("reltr_sgg_node")
    RelTRSGGNode()
    rospy.spin()

if __name__ == "__main__":
    main()