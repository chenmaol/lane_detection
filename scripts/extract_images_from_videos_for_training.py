video_path = 'wrcg_data/germany/videos/0.mp4'

import cv2
import os

# 创建保存图像的目录
save_dir = 'wrcg_data/germany/images/'
os.makedirs(save_dir, exist_ok=True)

# 读取视频
cap = cv2.VideoCapture(video_path)
frame_count = 0
n = 5
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # 每隔n帧保存一张图像
    if frame_count % n == 0:
        image_name = f'frame_{frame_count:06d}.jpg'
        save_path = os.path.join(save_dir, image_name)
        cv2.imwrite(save_path, frame)
        
    frame_count += 1

cap.release()
print(f'共处理了 {frame_count} 帧，保存了 {frame_count//n} 张图像')

