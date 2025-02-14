import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image
from screen import MyScreen
import win32gui
import time
from utils.utils import Action, SpeedRecognizer

def calculate_deviation(vehicle_position, lane_line):
    """
    计算车辆与可见车道线的偏差
    :param vehicle_position: 车辆的当前位置，形状为(2,)
    :param lane_line: 可见车道线的xy坐标，形状为(N, 2)
    :return: 车辆与车道线的横向偏差
    """
    # 找到最近的车道线点
    lane_line = np.array(lane_line)
    distances = np.linalg.norm(lane_line - vehicle_position, axis=1)
    closest_point_idx = np.argmin(distances)
    closest_point = lane_line[closest_point_idx]

    # 计算横向偏差（x方向）
    lateral_deviation = vehicle_position[0] - closest_point[0]
    return lateral_deviation


def decision(left_lane, right_lane, center_lane, vehicle_position=(640, 450)):
    # 判断可见车道线的位置
    if len(center_lane) >= 2:
        lateral_deviation = calculate_deviation(vehicle_position, center_lane)
    elif len(left_lane) >= 2:
        lateral_deviation = -300
    elif len(right_lane) >= 2:
        lateral_deviation = 300
    else:
        lateral_deviation = 0

    return lateral_deviation

def process_image(ori):
    img = Image.fromarray(ori[:, :, ::-1])
    img = img_transforms(img).unsqueeze(0).cuda()
    with torch.no_grad():
        out = net(img)

    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(cfg.griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == cfg.griding_num] = 0
    out_j = loc

    # import pdb; pdb.set_trace()
    vis = ori.copy()
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                           int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                    cv2.circle(vis, ppp, 5, (0, 255, 0), -1)
    return vis, out_j, col_sample_w


def calculate_steering(out_j, col_sample_w):
    """计算转向控制信号"""
    # 获取最近的车道线点(最下面一行)
    bottom_row = out_j[-10]
    
    # 找到左右车道线
    left_lane = None
    right_lane = None
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            if left_lane is None:
                left_lane = bottom_row[i] * col_sample_w
            else:
                right_lane = bottom_row[i] * col_sample_w
                break
    
    if left_lane is None or right_lane is None:
        return 0  # 如果检测不到车道线，保持直行
        
    # 计算车道中心
    lane_center = (left_lane + right_lane) / 2
    # 图像中心
    image_center = 400  # 800/2
    # 计算偏移量
    offset = lane_center - image_center
    
    # 简单的比例控制
    kp = 0.001  # 需要调整的参数
    steering = -kp * offset  # 负号是因为需要向偏移的反方向转向
    
    # 限制转向角度
    steering = max(-1.0, min(1.0, steering))
    return steering

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_w, img_h = 1280, 800
    row_anchor = culane_row_anchor

    hwnd = win32gui.FindWindow(None, "WRCG")
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(1)
    action = Action()
    action.press_key('esc')
    time.sleep(1)

    recorder = MyScreen((0, 0, 1280, 800))
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    writer = cv2.VideoWriter('output_vis.mp4', fourcc, 10.0, (img_w, img_h))
    # writer_ori = cv2.VideoWriter('output_ori.mp4', fourcc, 10.0, (img_w, img_h))
    speedRecognizer = SpeedRecognizer(img_size=(1280, 800))

    action_buffer = {'w': 0, 'a': 0, 'd': 0}
    buffer_frame_num = 1
    speed_frame_cnt = 0
    speed_frame_thres = 10
    steering_thres = [50, 300]
    try:
        while True:
            image = recorder.get_frame()
            # writer_ori.write(image)
            
            vis, out_j, col_sample_w = process_image(image)

            center_lane = []
            left_lane = []
            right_lane = []
            for i in range(len(out_j)):
                if out_j[i, 0] > 0:
                    ppp = (int(out_j[i, 0] * col_sample_w * img_w / 800) - 1,
                           int(img_h * (row_anchor[cls_num_per_lane - 1 - i] / 288)) - 1)
                    cv2.circle(vis, ppp, 5, (0, 0, 255), -1)
                    left_lane.append(ppp)
                if out_j[i, 1] > 0:
                    ppp = (int(out_j[i, 1] * col_sample_w * img_w / 800) - 1,
                           int(img_h * (row_anchor[cls_num_per_lane - 1 - i] / 288)) - 1)
                    cv2.circle(vis, ppp, 5, (255, 0, 0), -1)
                    right_lane.append(ppp)
                if out_j[i, 0] > 0 and out_j[i, 1] > 0:
                    center_j = out_j[i, 0] * 0.5 + out_j[i, 1] * 0.5
                    ppp = (int(center_j * col_sample_w * img_w / 800) - 1,
                           int(img_h * (row_anchor[cls_num_per_lane - 1 - i] / 288)) - 1)
                    cv2.circle(vis, ppp, 5, (0, 255, 0), -1)
                    center_lane.append(ppp)

            steering = decision(left_lane, right_lane, center_lane)

            speed = speedRecognizer.get_speed(image)
            # print(steering, speed)

            # 在图像上显示steering值
            steering_speed_text = f"Steering: {steering:.2f}, Speed: {speed}"
            cv2.putText(vis, steering_speed_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)

            writer.write(vis)

            if speed == 0:
                speed_frame_cnt += 1
                print(speed_frame_cnt)
                if speed_frame_cnt >= speed_frame_thres:
                    speed_frame_cnt = 0
                    for key in ['a', 'd', 'w']:
                        action.up_key(key)
                    action.press_key('s', 1)
                    action.press_key('w', 0.4)
            else:
                speed_frame_cnt = 0

            # break
            #根据转向控制发送指令
            for key in ['a', 'd', 'w']:
                action.up_key(key)

            if steering < -steering_thres[0]:
                if action_buffer['d'] <= 0:
                    action.down_key('d')
                    if steering < -steering_thres[1]:
                        action_buffer['d'] = 0
                    else:
                        action_buffer['d'] = buffer_frame_num
                else:
                    action_buffer['d'] -= 1
            elif steering > steering_thres[0]:
                if action_buffer['a'] <= 0:
                    action.down_key('a')
                    if steering > steering_thres[1]:
                        action_buffer['a'] = 0
                    else:
                        action_buffer['a'] = buffer_frame_num
                else:
                    action_buffer['a'] -= 1

            if speed < 20:
                if action_buffer['w'] <= 0:
                    action.down_key('w')
                    if speed < 18:
                        action_buffer['w'] = 0
                    else:
                        action_buffer['w'] = buffer_frame_num
                else:
                    action_buffer['w'] -= 1

    finally:
        # 确保在程序退出时释放所有按键
        for key in ['a', 'd', 'w']:
            action.up_key(key)
        writer.release()
        # writer_ori.release()



