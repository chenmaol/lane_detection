import json
import os
import base64

map = "wales"
# 读取 default.json 文件
with open(f'wrcg_data/{map}/annotations/default.json', 'r') as f:
    data = json.load(f)

# 创建输出目录
output_dir = f'wrcg_data/{map}/images'
os.makedirs(output_dir, exist_ok=True)

# 提取每个图像的标注信息
for item in data['items']:
    frame_id = item['id']
    annotations = item['annotations']
    image_info = item['image']

    # 构建新的 JSON 格式
    output_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_info['path'],
        "imageData": "",  # 将在后面填充
        "imageHeight": image_info['size'][0],
        "imageWidth": image_info['size'][1]
    }

    # 将标注信息转换为 shapes 格式
    for annotation in annotations:
        # 将 points 转换为每两个点组成一个点
        points = annotation['points']
        formatted_points = [points[i:i + 2] for i in range(0, len(points), 2)]

        shape = {
            "label": "left" if annotation['label_id'] == 0 else "right",
            "points": formatted_points,
            "group_id": None,
            "description": "",
            "shape_type": "linestrip",
            "flags": {}
        }
        output_data['shapes'].append(shape)

    # 读取图像文件并编码为 Base64
    image_path = os.path.join(output_dir, image_info['path'])
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
        output_data['imageData'] = image_data

    # 保存为单独的 JSON 文件
    output_file_path = os.path.join(output_dir, f"{frame_id}.json")
    with open(output_file_path, 'w') as output_file:
        json.dump(output_data, output_file, indent=2)

print("标注信息已成功提取并保存为单独的 JSON 文件。")