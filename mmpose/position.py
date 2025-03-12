import numpy as np
import json

def check_keypoint_reasonability(video_keypoints, video_keypoint_scores, confidence_threshold=0.3):
    """
    检查视频中关键点位置的合理性
    :param video_keypoints: 包含每帧关键点的列表
    :param video_keypoint_scores: 包含每帧关键点置信度得分的列表
    :param confidence_threshold: 关键点置信度的阈值
    :return: 异常检测结果
    """
    anomalies = []
    num_frames = len(video_keypoints)

    for frame in range(num_frames):
        keypoints = video_keypoints[frame]
        scores = video_keypoint_scores[frame]

        # 只考虑置信度高于阈值的关键点
        valid_keypoints = [kp for kp, score in zip(keypoints, scores) if score > confidence_threshold]
        
        if len(valid_keypoints) < 5:  # 确保有足够的有效关键点进行检查
            continue

        # 1. 检查头部位置（假设关键点0是鼻子）
        nose = np.array(valid_keypoints[0])
        left_shoulder = np.array(valid_keypoints[5])
        right_shoulder = np.array(valid_keypoints[6])
        left_hip = np.array(valid_keypoints[11])
        right_hip = np.array(valid_keypoints[12])

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        if nose[1] > shoulder_center[1]:  # y坐标越大，位置越低
            anomalies.append(f"Frame {frame}: Head is below shoulders")

        # 2. 检查肩膀位置相对于臀部的合理性
        if shoulder_center[1] > hip_center[1]:
            anomalies.append(f"Frame {frame}: Shoulders are below hips")

        # 3. 检查手臂长度的合理性
        left_elbow = np.array(valid_keypoints[7])
        left_wrist = np.array(valid_keypoints[9])
        right_elbow = np.array(valid_keypoints[8])
        right_wrist = np.array(valid_keypoints[10])

        upper_arm_length = np.linalg.norm(left_shoulder - left_elbow)
        forearm_length = np.linalg.norm(left_elbow - left_wrist)
        if forearm_length > 1.5 * upper_arm_length:
            anomalies.append(f"Frame {frame}: Left forearm is unusually long compared to upper arm")

        upper_arm_length = np.linalg.norm(right_shoulder - right_elbow)
        forearm_length = np.linalg.norm(right_elbow - right_wrist)
        if forearm_length > 1.5 * upper_arm_length:
            anomalies.append(f"Frame {frame}: Right forearm is unusually long compared to upper arm")

        # 4. 检查腿部长度的合理性
        left_knee = np.array(valid_keypoints[13])
        left_ankle = np.array(valid_keypoints[15])
        right_knee = np.array(valid_keypoints[14])
        right_ankle = np.array(valid_keypoints[16])

        thigh_length = np.linalg.norm(left_hip - left_knee)
        calf_length = np.linalg.norm(left_knee - left_ankle)
        if calf_length > 1.3 * thigh_length:
            anomalies.append(f"Frame {frame}: Left calf is unusually long compared to thigh")

        thigh_length = np.linalg.norm(right_hip - right_knee)
        calf_length = np.linalg.norm(right_knee - right_ankle)
        if calf_length > 1.3 * thigh_length:
            anomalies.append(f"Frame {frame}: Right calf is unusually long compared to thigh")

        # 5. 检查左右对称性
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        hip_width = np.linalg.norm(left_hip - right_hip)
        if hip_width > 1.5 * shoulder_width:
            anomalies.append(f"Frame {frame}: Hip width is unusually large compared to shoulder width")

    return anomalies

# 使用示例
with open('/mnt/workspace/lingxinran/metric/mmpose/vis_results/cogvideo/results_Cog5B_00019.json', 'r') as f:
    meta_info = json.load(f)
instance_info = meta_info['instance_info']
video_keypoints = [instance['instances'][0]['keypoints'] for instance in instance_info]
video_keypoint_scores = [instance['instances'][0]['keypoint_scores'] for instance in instance_info]
results = check_keypoint_reasonability(video_keypoints, video_keypoint_scores)
for result in results:
    print(result)
