#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
np.int = int   # 안전하게 alias
np.float = float

from PIL import Image

# ROS
import rospy
import rospkg
from std_msgs.msg import String
from cv_bridge import CvBridge

from google.genai import types
from google import genai

def get_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "nodes" not in data or not isinstance(data["nodes"], list):
        raise ValueError(f"Invalid JSON at {json_path}: missing 'nodes' list.")
    return data

def load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def clip_box_to_image(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))

    ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if ix2 <= ix1: ix2 = min(ix1 + 1, width)
    if iy2 <= iy1: iy2 = min(iy1 + 1, height)
    return ix1, iy1, ix2, iy2

def crop_image(img: Image.Image, bbox: List[float]) -> Image.Image:
    w, h = img.width, img.height
    ix1, iy1, ix2, iy2 = clip_box_to_image(tuple(bbox), w, h)
    return img.crop((ix1, iy1, ix2, iy2))

def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def get_caption_for_crop(jpeg_bytes: bytes, client: genai.Client, model_name: str) -> str:
    resp = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=jpeg_bytes, mime_type='image/jpeg'),
            "Briefly describe this object in the image."
        ]
    )
    return (resp.text or "").strip() if resp else ""


class CaptioningNode:
    def __init__(self):
        rospy.init_node("reltr_captioning_node", anonymous=True)
        self.bridge = CvBridge()

        pkg_root = Path(rospkg.RosPack().get_path('reltr_scene_graph'))
        self.pkg_root = pkg_root

        default_data_root = (pkg_root.parents[2] / 'data')
        self.data_root: Path = Path(rospy.get_param("~data_root", str(default_data_root))).resolve()

        default_sgg_route = (pkg_root.parent / 'reltr_scene_graph' / 'data')
        self.sgg_route: Path = Path(rospy.get_param("~sgg_route", str(default_sgg_route))).resolve()

        default_save_dir = (pkg_root.parents[2] / 'src' / 'captioning' / 'output')
        self.save_dir: Path = Path(rospy.get_param("~save_path", str(default_save_dir))).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        save_crops_dir = (self.save_dir / 'crops')
        self.save_crops_dir: Path = Path().resolve()
        if self.save_crops_dir:
            self.save_crops_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name: str = rospy.get_param("~model", "gemini-2.5-flash")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            rospy.logfatal("GEMINI_API_KEY is not set.")
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)

        self.if_process = False
        rospy.Subscriber("/reltr_mode", String, self.mode_callback)

        rospy.loginfo(f"[CaptioningNode] data_root={self.data_root}")
        rospy.loginfo(f"[CaptioningNode] sgg_route={self.sgg_route}")
        rospy.loginfo(f"[CaptioningNode] save_dir={self.save_dir}")
        rospy.loginfo(f"[CaptioningNode] model={self.model_name}")
    
    def mode_callback(self, msg: String):
        if msg.data != "fin":
            self.if_process = False
            return
        if not self.if_process:
            self.if_process = True
            try:
                self.extract_all_captions()
            except Exception as e:
                rospy.logerr(f"extract_all_captions failed: {e}")
            finally:
                self.if_process = False

    def extract_all_captions(self):
        merged_dir = self.sgg_route / "merged_sg"
        if not merged_dir.is_dir():
            raise FileNotFoundError(f"merged_sg dir not found: {merged_dir}")
        
        node_ids = []
        for name in os.listdir(self.data_root):
            try:
                node_ids.append(int(name))
            except Exception:
                pass
        node_ids = sorted(node_ids)

        for node_idx in node_ids:
            try:
                json_path = merged_dir / f"merged_sg_{node_idx}.json"
                if not json_path.is_file():
                    rospy.logwarn(f"JSON not found for node {node_idx}: {json_path}")
                    continue

                sg = get_json(str(json_path))
                updated = self.crop_and_caption_one_node(node_idx, sg)

                save_path = self.save_dir / f"merged_sg_{node_idx}_with_captions.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(updated, f, ensure_ascii=False, indent=2)
                rospy.loginfo(f"[OK] Saved updated JSON -> {save_path}")

            except Exception as e:
                rospy.logerr(f"[node {node_idx}] failed: {e}")

    def crop_and_caption_one_node(self, node_idx: int, sg: Dict[str, Any]) -> Dict[str, Any]:
        img_dir = self.data_root / str(node_idx) / "image"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"image dir not found: {img_dir}")

        for i, node in enumerate(sg.get("nodes", [])):
            bbox = None
            stamp = None

            bboxes = node.get("bboxes", {})
            if isinstance(bboxes, dict) and len(bboxes) > 0:
                stamp = sorted(bboxes.keys())[0]
                bbox = bboxes[stamp]
            elif "bbox" in node and isinstance(node["bbox"], (list, tuple)) and len(node["bbox"]) == 4:
                bbox = node["bbox"]

            if bbox is None:
                continue
            
            if stamp is not None:
                image_path = img_dir / f"{stamp}.png"
            else:
                candidates = sorted([p for p in img_dir.glob("*.png")])
                if not candidates:
                    rospy.logwarn(f"[node {node_idx}] no images in {img_dir}")
                    continue
                image_path = candidates[0]

            if not image_path.is_file():
                rospy.logwarn(f"[node {node_idx}] image not found: {image_path}")
                continue

            img = load_image(str(image_path))
            cropped = crop_image(img, bbox)
            jpeg_bytes = pil_to_jpeg_bytes(cropped, quality=90)

            try:
                caption = get_caption_for_crop(jpeg_bytes, self.client, self.model_name)
            except Exception as e:
                rospy.logwarn(f"[node {node_idx}] captioning failed at node[{i}] ({image_path.name}): {e}")
                caption = ""

            node["caption"] = caption

            if self.save_crops_dir:
                crop_path = self.save_crops_dir / f"{node_idx}" / f"crop_{i:04d}.jpg"
                try:
                    cropped.save(str(crop_path), "JPEG", quality=90)
                except Exception as e:
                    rospy.logwarn(f"failed to save crop: {crop_path} ({e})")
        
        return sg


if __name__ == "__main__":
    try:
        node = CaptioningNode()
        rospy.spin()
    except Exception as e:
        rospy.logfatal(f"Fatal error: {e}")
        raise
