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
from collections import deque
import math
from utils.utils import Action, ActionThread, SpeedRecognizer

class PIDSteerControl:
    def __init__(self, Kp=0.2, Ki=0.05, Kd=0.25, dt=0.1):
        self.Kd = Kd
        self.Ki = Ki
        self.Kp = Kp
        self.dt = dt
        self.errorBuffer = deque(maxlen = int(0.25 / dt))
        self.last_steering = 0  # 添加上一次转向记录
        self.last_waypoint = None

    # Getting velocities
    def pid_controller(self, waypoint):
        # if self.last_waypoint:
        #     waypoint = [waypoint[0] * 0.5 + self.last_waypoint[0] * 0.5, waypoint[1] * 0.5 + self.last_waypoint[1] * 0.5]
        #
        # self.last_waypoint = waypoint

        v_begin = np.array([0.5, 1.0])
        v_end = np.array([0.5, 0.5])
        v_vec = v_end - v_begin
        w_vec = np.array([waypoint[0] - v_begin[0], waypoint[1] - v_begin[1]])
        dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        cross_product_2d = v_vec[0] * w_vec[1] - v_vec[1] * w_vec[0]

        if cross_product_2d < 0:
            dot *= -1

        # 添加死区
        deadzone = 0.03
        if abs(dot) < deadzone:
            dot = 0
        self.errorBuffer.append(dot)

        # Calculating errors:
        if len(self.errorBuffer) >= 2:
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt
        else:
            de = 0.0
            ie = 0.0
        steering =  np.clip((self.Kp * dot) + (self.Ki * ie) + (self.Kd * de), -1.0, 1.0)
        # 平滑转向变化
        # steering = 0.5 * steering + 0.5 * self.last_steering
        self.last_steering = steering
        return steering

    def reset(self):
        self.errorBuffer.clear()

class PIDSpeedControl:
    def __init__(self, Kp=0.25, Ki=0.05, Kd=0.25, dt=0.1):
        self.Kd = Kd
        self.Ki = Ki
        self.Kp = Kp
        self.dt = dt
        self.errorBuffer = deque(maxlen = int(0.25 / dt))

    # Getting velocities
    def pid_controller(self, target_speed, current_speed):
        error = (target_speed - current_speed) / target_speed
        self.errorBuffer.append(error)

        if len(self.errorBuffer) >= 2:
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt
        else:
            de = 0.0
            ie = 0.0
        return np.clip(self.Kp * error + self.Kd * de + self.Ki * ie, -1.0, 1.0)

    def reset(self):
        self.errorBuffer.clear()

def process_image(ori):
    t0 = time.time()
    img = ori.copy()
    img = cv2.resize(img, (800, 288))  # 替代 transforms.Resize

    # 归一化和转换为tensor
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = torch.from_numpy(img).float().unsqueeze(0).cuda()
    t1 = time.time()
    with torch.no_grad():
        out = net(img)
    t2 = time.time()
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
    t3 = time.time()
    # import pdb; pdb.set_trace()
    vis = ori[:, :, ::-1].copy()
    # for i in range(out_j.shape[1]):
    #     if np.sum(out_j[:, i] != 0) > 2:
    #         for k in range(out_j.shape[0]):
    #             if out_j[k, i] > 0:
    #                 ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
    #                        int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
    #                 cv2.circle(vis, ppp, 5, (0, 255, 0), -1)
    t4 = time.time()
    # print("process, ", t1 - t0, t2 - t1, t3 - t2, t4 - t3)
    return vis, out_j, col_sample_w


def decision(left_lane, right_lane, center_lane, speed):
    target_speed = 40
    # 判断可见车道线的位置
    if len(center_lane) >= 2:
        waypoint = center_lane[-1]
    elif len(left_lane) >= 2:
        waypoint = left_lane[-1]
        waypoint = [waypoint[0] + img_w * 0.2, waypoint[1]]
        target_speed *= 0.5
    elif len(right_lane) >= 2:
        waypoint = right_lane[-1]
        waypoint = [waypoint[0] - img_w * 0.2, waypoint[1]]
        target_speed *= 0.3
    else:
        waypoint = None

    if waypoint:

        steering = steer_controller.pid_controller([waypoint[0] / img_w, waypoint[1] / img_h])
    else:
        steering = 0

    steer_thres = 0.03
    for key in ['a', 'd']:
        action.up_key(key)

    if steering < -steer_thres:
        # action.up_key('d')
        action.down_key('a')
    elif steering > steer_thres:
        # action.up_key('a')
        action.down_key('d')

    gas = speed_controller.pid_controller(target_speed, speed)
    print(gas)
    if gas > 0.3:
        action.up_key('s')
        action.down_key('w')
    elif gas < -0.3:
        action.up_key('w')
        action.down_key('s')
    # elif gas < -0.0:
    #     action.up_key('w')
    #     action.up_key('s')
    # gas = 0
    # if speed < target_speed:
    #     action.down_key('w')
    #     action.up_key('s')
    #     gas = 1
    # elif target_speed * 1.2 > speed > target_speed:
    #     action.up_key('w')
    #     action.up_key('s')
    # elif speed > target_speed * 1.2:
    #     action.up_key('w')
    #     action.down_key('s')
    #     gas = -1

    return steering, gas


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()  # we dont need auxiliary segmentation in testing

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
    img_w, img_h = 1920, 1080
    fps_rate = 0.04
    row_anchor = culane_row_anchor
    last_waypoint = None

    hwnd = win32gui.FindWindow(None, "WRCG")
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(1)
    action = Action()
    action.press_key('esc')
    time.sleep(1)

    recorder = MyScreen()
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter('output_vis.mp4', fourcc, 1 / fps_rate, (img_w, img_h))
    # writer_ori = cv2.VideoWriter('output_ori.mp4', fourcc, 10.0, (img_w, img_h))
    speedRecognizer = SpeedRecognizer(img_size=(img_w, img_h))
    steer_controller = PIDSteerControl(dt=fps_rate)
    speed_controller = PIDSpeedControl(dt=fps_rate)

    speed_frame_cnt = 0
    speed_frame_thres = 1 / fps_rate


    try:
        while True:
            t0 = time.time()
            image = recorder.get_frame()
            t1 = time.time()
            # image = cv2.resize(image, (1280, 800))
            # writer_ori.write(image)

            vis, out_j, col_sample_w = process_image(image)
            t2 = time.time()

            center_lane = []
            left_lane = []
            right_lane = []
            for i in range(len(out_j)):  # [18, 2]
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

            speed = speedRecognizer.get_speed(image)
            steering, gas = decision(left_lane, right_lane, center_lane, speed)

            # 在图像上显示steering值
            steering_speed_text = f"Steering: {steering:.2f}, Gas: {gas: .2f}"
            cv2.putText(vis, steering_speed_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)

            writer.write(vis)

            if speed == 0:
                speed_frame_cnt += 1
                if speed_frame_cnt >= speed_frame_thres:
                    speed_frame_cnt = 0
                    for key in ['a', 'd', 'w', 's']:
                        action.up_key(key)
                    action.press_key('s', 1)
                    action.press_key('w', 0.4)
                    speed_controller.reset()
                    steer_controller.reset()
            else:
                speed_frame_cnt = 0

            while time.time() - t0 < fps_rate:
                pass
            # print(time.time() - t0, t1 - t0, t2 - t1)

    finally:
        # 确保在程序退出时释放所有按键
        for key in ['a', 'd', 'w', 's']:
            action.up_key(key)
        writer.release()
        # writer_ori.release()



