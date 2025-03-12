import numpy as np
import json
from scipy.spatial import distance

def calculate_angle(point1, point2, point3):
    """计算三个点形成的角度"""
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def analyze_joint_angles(video_keypoints, video_keypoint_scores, threshold=30, confidence_threshold=0.3, window_size=1):
    """
    分析视频中关节角度的变化，使用以帧为中心的滑动窗口
    :param video_keypoints: 包含每帧关键点的列表
    :param video_keypoint_scores: 包含每帧关键点置信度得分的列表
    :param threshold: 允许的最大角度变化（度）
    :param confidence_threshold: 关键点置信度的阈值
    :param window_size: 滑动窗口的大小（应为奇数）
    :return: 异常检测结果
    """
    num_frames = len(video_keypoints)
    if num_frames < window_size:
        return "Not enough frames for analysis"

    # 确保窗口大小是奇数
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    half_window = window_size // 2

    # 定义需要分析的关节角度
    joint_angles = [
        ('left_elbow', 5, 7, 9),    # 左肩-左肘-左手腕
        ('right_elbow', 6, 8, 10),  # 右肩-右肘-右手腕
        ('left_shoulder', 7, 5, 11), # 左肘-左肩-左臀
        ('right_shoulder', 8, 6, 12), # 右肘-右肩-右臀
        ('left_hip', 5, 11, 13),    # 左肩-左臀-左膝
        ('right_hip', 6, 12, 14),   # 右肩-右臀-右膝
        ('left_knee', 11, 13, 15),  # 左臀-左膝-左踝
        ('right_knee', 12, 14, 16)  # 右臀-右膝-右踝
    ]

    angles = {angle[0]: [] for angle in joint_angles}
    valid_measurements = {angle[0]: [] for angle in joint_angles}

    # 计算每帧的角度，同时考虑置信度
    for frame_keypoints, frame_scores in zip(video_keypoints, video_keypoint_scores):
        for angle_name, p1, p2, p3 in joint_angles:
            if all(frame_scores[i] > confidence_threshold for i in [p1, p2, p3]):
                angle = calculate_angle(frame_keypoints[p1], frame_keypoints[p2], frame_keypoints[p3])
                angles[angle_name].append(angle)
                valid_measurements[angle_name].append(True)
            else:
                angles[angle_name].append(None)
                valid_measurements[angle_name].append(False)


    for i, angle in enumerate(angles['right_elbow']):
        print(f"Frame {i}: Right elbow angle = {angle}")

    # 分析角度的变化，使用以帧为中心的滑动窗口
    anomalies = []
    for angle_name in angles:
        valid_angles = [a for a, v in zip(angles[angle_name], valid_measurements[angle_name]) if v and a is not None]
        
        if len(valid_angles) < num_frames // 3:
            anomalies.append(f"{angle_name} has insufficient valid measurements")
            continue
        
        window_averages = []
        for i in range(len(valid_angles)):
            start = max(0, i - half_window)
            end = min(len(valid_angles), i + half_window + 1)
            window = valid_angles[start:end]
            window_averages.append(np.mean(window))
        
        # 计算相邻窗口平均值的变化
        changes = np.abs(np.diff(window_averages))
        
        for i, change in enumerate(changes):
            if change > threshold:
                anomalies.append(f"{angle_name} shows significant change around frame {i}, change: {change:.2f} degrees")
                break

    return anomalies

# 使用示例
with open('/mnt/workspace/lingxinran/metric/mmpose/vis_results/cogvideo/results_Cog5B_00019.json', 'r') as f:
    meta_info = json.load(f)
instance_info = meta_info['instance_info']
video_keypoints = [instance['instances'][0]['keypoints'] for instance in instance_info]
video_keypoint_scores = [instance['instances'][0]['keypoint_scores'] for instance in instance_info]
results = analyze_joint_angles(video_keypoints, video_keypoint_scores)
for result in results:
    print(result)
