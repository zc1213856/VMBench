import numpy as np
import json
import matplotlib.pyplot as plt

def visualize_score_changes(video_keypoint_scores, visibility_threshold=0.3):
    """
    可视化每帧关键点得分的平均变化
    :param video_keypoint_scores: 包含每帧关键点置信度得分的列表
    :param visibility_threshold: 关键点被认为是可见的最小得分阈值
    """
    # 将得分转换为numpy数组
    scores = np.array(video_keypoint_scores)
    num_frames, num_keypoints = scores.shape

    # 创建一个布尔掩码，标记每帧中的可见关键点
    visible_mask = scores > visibility_threshold

    # 计算每帧可见关键点的平均得分
    frame_mean_scores = np.sum(scores * visible_mask, axis=1) / np.sum(visible_mask, axis=1)

    # 处理可能的除零情况
    frame_mean_scores = np.nan_to_num(frame_mean_scores, nan=0)

    # 计算整个视频的全局平均得分
    global_mean = np.mean(frame_mean_scores)

    # 创建图表
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_frames), frame_mean_scores, label='Frame Average Score')
    plt.axhline(y=global_mean, color='r', linestyle='--', label='Global Average Score')

    plt.title('Average Keypoint Score per Frame')
    plt.xlabel('Frame Number')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid(True)

    # 添加全局平均分数的标注
    plt.text(num_frames * 0.02, global_mean, f'Global Average: {global_mean:.2f}', 
             verticalalignment='bottom', horizontalalignment='left')

    plt.tight_layout()
    plt.savefig('keypoint_score_changes.png')

# 使用示例
with open('/mnt/workspace/lingxinran/metric/mmpose/vis_results/cogvideo/results_Cog5B_00016.json', 'r') as f:
    meta_info = json.load(f)
instance_info = meta_info['instance_info']
video_keypoints = [instance['instances'][0]['keypoints'] for instance in instance_info]
video_keypoint_scores = [instance['instances'][0]['keypoint_scores'] for instance in instance_info]

visualize_score_changes(video_keypoint_scores)