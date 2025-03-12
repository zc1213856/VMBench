![title.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJMJEg6rjq3p8/img/666040a8-4ec1-474f-840e-f039539d7d08.png)

# 🔥 Updates

*   \[3/2024\] **VMBench** evaluation code & prompt set released!
    

# 📣 Overview

![overview.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJMJEg6rjq3p8/img/27919c5f-fcea-4f50-8ed8-835ede58b392.png)

Video generation has advanced rapidly, improving evaluation methods, yet assessing video's motion remains a major challenge. Specifically, there are two key issues: 1) current motion metrics do not fully align with human perceptions; 2) the existing motion prompts are limited. Based on these findings, we introduce **VMBench**---a comprehensive **V**ideo **M**otion **Bench**mark that has perception-aligned motion metrics and features the most diverse types of motion. VMBench has several appealing properties: (1) **Perception-Driven Motion Evaluation Metrics**, we identify five dimensions based on human perception in motion video assessment and develop fine-grained evaluation metrics, providing deeper insights into models' strengths and weaknesses in motion quality. (2) **Meta-Guided Motion Prompt Generation**, a structured method that extracts meta-information, generates diverse motion prompts with LLMs, and refines them through human-AI validation, resulting in a multi-level prompt library covering six key dynamic scene dimensions. (3) **Human-Aligned Validation Mechanism**, we provide human preference annotations to validate our benchmarks, with our metrics achieving an average 35.3% improvement in Spearman’s correlation over baseline methods. This is the first time that the quality of motion in videos has been evaluated from the perspective of human perception alignment. Additionally, we will soon release VMBench as an open-source benchmark, setting a new standard for evaluating and advancing motion generation models.

# 📊Evaluation Results

## Gallery

**Prompt:** A tourist joyfully splashes water in an outdoor swimming pool, their arms and legs moving energetically as they playfully splash around.
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/74a0f3b0-6a39-42fe-98de-4a18d4130837" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7437404c-d732-4e57-9b74-bf1977bc5bfc" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/fbce3824-2cab-426f-a684-ba020366fea2" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/adf91760-ee43-4dae-8675-d6be9584ef98" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/095d2455-2456-4fdf-a36f-1eee7d3485df" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/14b8787b-1fe0-4c75-9c93-0107198a436f" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

**Prompt:** Three books are thrown into the air, their pages fluttering as they soar over the soccer field, landing in a scattered pattern.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/5e5432c0-4eb1-40cb-b38d-cb179763d3f8" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/dba7d40c-de5e-4f58-aa2c-4359daa0b358" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/6d3fd125-52e9-482a-b298-a457b5dcf0c9" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/114c250c-02b1-4755-8f5a-7ab0c2531c6f" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/e4bd1e41-6fc4-48ab-87a1-cb50788a8041" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/ab7e94d4-69c0-47bf-9868-fbd5bcad9b1c" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

**Prompt:** Four flickering candles cast shadows as they burn steadily on the balcony, their flames dancing with the gentle breeze.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/791fd2ad-0604-4c09-9dde-1eaf5254ea4e" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b5eca664-8ff2-41aa-9f77-0c30bc0b197f" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/f15f08cd-67f0-4053-a571-6a4afd39cd11" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3d6006fc-f70a-4b28-a789-5a8d156e5368" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/1c98110c-0372-41ea-864a-57f5f7fdd5c7" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/2aa00829-1894-49b7-8572-42364dc890e5" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

**Prompt:** Two penguins waddle along the beach, occasionally stopping to preen their feathers before continuing their journey across the ocean shore.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/448a733c-4a76-4186-bf6e-a9c0b0515658" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ee0c4949-319b-4e6b-a00f-b36d5bffd3f5" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/7089fd0a-23b8-454a-9c75-330ebafa9eb0" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/5bf476c1-707d-45fb-b77e-922b776be221" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/396ac6f7-9073-45b3-9fb1-42e896aae481" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/a92b2a48-b35d-412a-8472-49eca441fbfb" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

**Prompt:** In the bustling street, two kids run towards a small dog, bending down to carefully comb its fur, their hands moving swiftly.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/235fa02e-0e65-4e23-bfe3-5cfc2f82d7e2" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d13fd47c-3bfc-44a6-930c-4108e006f9d6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/9aaeca69-2e8b-461b-8934-d96eba0a5c2c" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/4bb5d304-4e6e-44c4-91b8-81c44513087f" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/30aaab94-bd69-49b1-9334-e44e129c4ee8" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/d701dfcb-aad9-4e38-8ff7-437c57374a31" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

**Prompt:** In the garage, a young girl twirls gracefully, her arms outstretched, perfectly matching the lively country line dance beat.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/7f379bf0-648d-4fc5-b267-2148b1959eef" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d371ec05-4c2c-496e-a5be-c8e84229cc5c" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/09275d8a-8313-4174-bdaa-448984e3dab0" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/837d5a79-d144-469e-9e6a-0d32c129086f" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/103de7c4-608c-4434-a065-c18b74e0e421" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/33a152e6-bc5d-459f-9587-ea2c801dec90" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

## Quantitative Results

*   [ ] 模型表现的五边形图
    
### VMBench Leaderboard

# 🔨 Installation

## Create Environment

```shell
git clone https://github.com/Ran0618/VMBench.git
cd VMBench

# create conda environment
conda create -n VMBench python=3.10
pip install torch torchvision

# Install Grounded-Segment-Anything module
cd Grounded-Segment-Anything
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install -r requirements.txt

# Install Groudned-SAM-2 module
cd Grounded-SAM-2
pip install -e .

# Install Q-Align module
cd Q-Align
pip install -e .

# Install VideoMAEv2 module
cd VideoMAEv2
pip install -r requirements.txt
```

## Download checkpoints
Place the pre-trained checkpoint files in the `.cache` directory at the root of the repository.
Our model's checkpoints are provided on HuggingFace.

```shell
mkdir .cache
cd .cache

huggingface-cli download [your-org]/[your-model] --local-dir .cache/
```
Please organize the pretrained models in this structure:
```shell
VMBench/.cache
├── baseline_offline.pth
├── baseline_online.pth
├── groundingdino_swinb_cogcoor.pth
├── groundingdino_swint_ogc.pth
├── sam2.1_hiera_large.pt
├── sam_vit_h_4b8939.pth
├── scaled_offline.pth
├── scaled_online.pth
└── vit_g_vmbench.pt
```

# 🔧Usage

## Videos Preparation

Generate videos of your model using the 1050 prompts provided in `prompts/prompts.txt` or `prompts/prompts.json` and organize them in the following structure:

```shell
VMBench/eval_results/videos
├── 0001.mp4
├── 0002.mp4
...
└── 1050.mp4
```

**Note:** Ensure that you maintain the correspondence between prompts and video sequence numbers. The index for each prompt can be found in the `prompts/prompts.json` file.

You can follow us `sample_video_demo.py` to generate videos. Or you can put the results video named index into your own folder.
    

## Evaluation on the VMBench
To evaluate generated videos using the VMBench, run the following command:

```shell
bash evaluate.sh your_videos_folder
```

The evaluation results for each video will be saved in the `./eval_results/${current_time}/results.json`. Scores for each dimension and the total score will be stored in the `./eval_results/${current_time}/scores.csv`.

# ❤️Acknowledgement
We would like to express our gratitude to the following open-source repositories that our work is based on: [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [GroundedSAM2](https://github.com/IDEA-Research/Grounded-SAM-2), [Co-Tracker](https://github.com/facebookresearch/co-tracker), [MMPose](https://github.com/open-mmlab/mmpose), [Q-Align](https://github.com/Q-Future/Q-Align), [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2), [VideoAlign](https://github.com/KwaiVGI/VideoAlign).
Their contributions have been invaluable to this project.

# 📜License
The VMBench is licensed under [Apache-2.0 license](http://www.apache.org/licenses/LICENSE-2.0). You are free to use our codes for research purpose.

# ✏️Citation
