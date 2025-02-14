import cv2
import numpy as np
from PIL import Image

def visualize_labels(image_path):
    # 读取图片
    img = cv2.imread(image_path)
    # img = Image.open(image_path)
    # img.show()
    if img is None:
        raise ValueError("无法读取图片")

    # 创建一个与原图相同大小的彩色图像
    colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    # 定义颜色映射 (BGR格式)
    color_map = {
        0: [0, 0, 0],      # 黑色
        1: [0, 255, 0],    # 绿色
        2: [0, 0, 255]     # 红色
    }
    
    # 对每个像素值进行颜色映射
    for label, color in color_map.items():
        colored[img == label] = color
    
    # 显示图片
    cv2.imshow('Visualized Labels', colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 替换为你的图片路径
    image_path = "../wrcg_data/labels_dir/frame_000279.png"
    visualize_labels(image_path)
