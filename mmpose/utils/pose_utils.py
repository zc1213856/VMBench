import json
import numpy as np
from scipy.spatial import distance
from scipy.signal import medfilt
from scipy.stats import ttest_ind

def analyze_lengths_over_time(instance_info, threshold=0.45, min_length=1e-6, confidence_threshold=0.3, window_size=1):
    video_keypoints = [instance['instances'][0]['keypoints'] for instance in instance_info]
    video_keypoint_scores = [instance['instances'][0]['keypoint_scores'] for instance in instance_info]

    num_frames = len(video_keypoints)
    if num_frames < window_size:
        return "Not enough frames for analysis", 0  # 返回错误消息和0分

    # 确保窗口大小是奇数
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    half_window = window_size // 2

    body_parts = [
        ('torso', 5, 11),
        ('left_upper_arm', 5, 7),
        ('left_forearm', 7, 9),
        ('right_upper_arm', 6, 8),
        ('right_forearm', 8, 10),
        ('left_thigh', 11, 13),
        ('left_calf', 13, 15),
        ('right_thigh', 12, 14),
        ('right_calf', 14, 16),
        ('shoulder_width', 5, 6),
        ('hip_width', 11, 12),
        ('neck', 0, 5),
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
    normal_parts = 0  # 计数正常的身体部位
    total_parts = len(body_parts)  # 总的身体部位数量

    for part_name in lengths:
        valid_lengths = [l for l, v in zip(lengths[part_name], valid_measurements[part_name]) if v and l is not None and l > min_length]
        
        if len(valid_lengths) < num_frames // 2:
            # 如果没有足够的有效测量点，我们认为这个部位是正常的
            normal_parts += 1
            continue
        
        window_averages = []
        for i in range(len(valid_lengths)):
            start = max(0, i - half_window)
            end = min(len(valid_lengths), i + half_window + 1)
            window = valid_lengths[start:end]
            window_averages.append(np.mean(window))
        
        # 计算相邻窗口平均值的相对变化
        part_normal = True
        for i in range(len(window_averages) - 1):
            change = abs(window_averages[i+1] - window_averages[i]) / window_averages[i]
            if change > threshold:
                anomalies.append(f"{part_name} shows significant change around frame {i}, change: {change:.2%}")
                part_normal = False
                break
        
        if part_normal:
            normal_parts += 1

    # 计算得分
    score = round(normal_parts / total_parts * 100, 2)

    return anomalies, score
