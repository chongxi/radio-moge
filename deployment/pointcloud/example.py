"""python example.py frame.png --output frame.ply --device cuda"""

import argparse
from pathlib import Path

from PIL import Image
import torch

from radio_moge import PointCloudPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--weights", default=str(Path(__file__).resolve().parent / "weights"))
    parser.add_argument("--output", default="frame.ply")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)  # Application policy; the library never changes this global setting.
    pipeline = PointCloudPipeline.from_pretrained(args.weights, device=args.device)
    # Keep pipeline alive and reuse it for subsequent frames.
    with Image.open(args.image) as image:
        result = pipeline(image)
    xyz, rgb = result.pointcloud(numpy=True)
    result.save_ply(args.output)
    print(f"depth={tuple(result.depth.shape)}, XYZ={xyz.shape}, RGB={rgb.shape}, PLY={args.output}")


if __name__ == "__main__":
    main()
