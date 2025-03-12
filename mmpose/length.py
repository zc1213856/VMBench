import json
import numpy as np
from scipy.spatial import distance
from scipy.signal import medfilt
from scipy.stats import ttest_ind

def analyze_body_proportions_over_time(video_keypoints, video_keypoint_scores, threshold=0.45, min_length=1e-6, confidence_threshold=0.3, window_size=1):
    """
    分析视频中人体各部分长度的变化，使用以帧为中心的滑动窗口
    :param video_keypoints: 包含每帧关键点的列表
    :param video_keypoint_scores: 包含每帧关键点置信度得分的列表
    :param threshold: 允许的最大变化比例
    :param min_length: 考虑有效的最小长度
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

    body_parts = [
        ('torso', 5, 11),       # 左肩到左臀
        ('left_upper_arm', 5, 7),  # 左肩到左肘
        ('left_forearm', 7, 9),    # 左肘到左手腕
        ('right_upper_arm', 6, 8), # 右肩到右肘
        ('right_forearm', 8, 10),  # 右肘到右手腕
        ('left_thigh', 11, 13),    # 左臀到左膝
        ('left_calf', 13, 15),     # 左膝到左脚踝
        ('right_thigh', 12, 14),   # 右臀到右膝
        ('right_calf', 14, 16),    # 右膝到右脚踝
        ('shoulder_width', 5, 6),  # 左肩到右肩
        ('hip_width', 11, 12),     # 左臀到右臀
        ('neck', 0, 5),            # 鼻子到左肩（近似颈部）
    ]

    lengths = {part[0]: [] for part in body_parts}
    valid_measurements = {part[0]: [] for part in body_parts}

    # 计算每帧的长度，同时考虑置信度
    for frame_keypoints, frame_scores in zip(video_keypoints, video_keypoint_scores):
        for part_name, start_idx, end_idx in body_parts:
            start_point = np.array(frame_keypoints[start_idx])
            end_point = np.array(frame_keypoints[end_idx])
            start_score = frame_scores[start_idx]
            end_score = frame_scores[end_idx]
            
            if start_score > confidence_threshold and end_score > confidence_threshold:
                length = distance.euclidean(start_point, end_point)
                lengths[part_name].append(length)
                valid_measurements[part_name].append(True)
            else:
                lengths[part_name].append(None)
                valid_measurements[part_name].append(False)

    # 分析长度的变化，使用以帧为中心的滑动窗口
    anomalies = []
    for part_name in lengths:
        valid_lengths = [l for l, v in zip(lengths[part_name], valid_measurements[part_name]) if v and l is not None and l > min_length]
        
        if len(valid_lengths) < num_frames // 2:
            anomalies.append(f"{part_name} has insufficient valid measurements")
            continue
        
        window_averages = []
        for i in range(len(valid_lengths)):
            start = max(0, i - half_window)
            end = min(len(valid_lengths), i + half_window + 1)
            window = valid_lengths[start:end]
            window_averages.append(np.mean(window))
        
        # 计算相邻窗口平均值的相对变化
        for i in range(len(window_averages) - 1):
            change = abs(window_averages[i+1] - window_averages[i]) / window_averages[i]
            if change > threshold:
                anomalies.append(f"{part_name} shows significant change around frame {i}, change: {change:.2%}")
                break

    return anomalies

# 使用示例
with open('/mnt/workspace/lingxinran/metric/mmpose/vis_results/cogvideo/results_Cog5B_00000.json', 'r') as f:
    meta_info = json.load(f)
instance_info = meta_info['instance_info']
video_keypoints = [instance['instances'][0]['keypoints'] for instance in instance_info]
video_keypoint_scores = [instance['instances'][0]['keypoint_scores'] for instance in instance_info]
results = analyze_body_proportions_over_time(video_keypoints, video_keypoint_scores)
for result in results:
    print(result)
