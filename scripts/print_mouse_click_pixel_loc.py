import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        print(f'点击位置坐标: x={x / w:.4f}, y={y / h:.4f}')

# 创建一个窗口
cv2.namedWindow('image', 0)
# 设置鼠标回调函数
cv2.setMouseCallback('image', mouse_callback)

# 读取图片（这里用一个空白图片作为示例，你可以替换成你自己的图片路径）
# 创建一个白色背景的图片
image = cv2.imread('../Screenshot 2025-01-07 210331.png')
h, w = image.shape[:2]
while True:
    cv2.imshow('image', image)
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
