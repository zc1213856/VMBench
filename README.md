# VMBench README

![title.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJMJEg6rjq3p8/img/666040a8-4ec1-474f-840e-f039539d7d08.png)

# 🔥 Updates

*   \[3/2024\] **VMBench** evaluation code & prompt set released!
    

# 📣 Overview

![overview.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/4j6OJMJEg6rjq3p8/img/27919c5f-fcea-4f50-8ed8-835ede58b392.png)

Video generation has advanced rapidly, improving evaluation methods, yet assessing video's motion remains a major challenge. Specifically, there are two key issues: 1) current motion metrics do not fully align with human perceptions; 2) the existing motion prompts are limited. Based on these findings, we introduce **VMBench**\---a comprehensive **V**ideo **M**otion **Bench**mark that has perception-aligned motion metrics and features the most diverse types of motion. VMBench has several appealing properties: (1) **Perception-Driven Motion Evaluation Metrics**, we identify five dimensions based on human perception in motion video assessment and develop fine-grained evaluation metrics, providing deeper insights into models' strengths and weaknesses in motion quality. (2) **Meta-Guided Motion Prompt Generation**, a structured method that extracts meta-information, generates diverse motion prompts with LLMs, and refines them through human-AI validation, resulting in a multi-level prompt library covering six key dynamic scene dimensions. (3) **Human-Aligned Validation Mechanism**, we provide human preference annotations to validate our benchmarks, with our metrics achieving an average 35.3\% improvement in Spearman’s correlation over baseline methods. This is the first time that the quality of motion in videos has been evaluated from the perspective of human perception alignment. Additionally, we will soon release VMBench as an open-source benchmark, setting a new standard for evaluating and advancing motion generation models.

# 📊Evaluation Results

## Gallery

**Prompt:** A tourist joyfully splashes water in an outdoor swimming pool, their arms and legs moving energetically as they playfully splash around.

| **CogVideoX-5B**                                             | [请至钉钉文档查看附件《A tourist joyfully splashes water in an outdoor swimming pool, their arms and legs moving energetica\_hunyuan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lj7736rnh8mlzz51&utm_medium=im_card&utm_scene=person_space&utm_source=im) **HunyuanVideo** | [请至钉钉文档查看附件《A tourist joyfully splashes water in an outdoor swimming pool, their arms and legs moving energetica\_mochi.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lj7e37zf12f8eptb&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Mochi 1** |
| :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [请至钉钉文档查看附件《A tourist joyfully splashes water in an outdoor swimming pool, their arms and legs moving energetica\_opensora\_plan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lj7sdhimszzu4qwed&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-Plan-v1.3.0** | [请至钉钉文档查看附件《A tourist joyfully splashes water in an outdoor swimming pool, their arms and legs moving energetica\_opensora.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lj8dpqt908qe1yzmi&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-v1.2** | [请至钉钉文档查看附件《A tourist joyfully splashes water in an outdoor swimming pool, their arms and legs moving energetica.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7mxznpr6mmodztdutt&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Wan2.1** |

**Prompt:** Three books are thrown into the air, their pages fluttering as they soar over the soccer field, landing in a scattered pattern.

|  [请至钉钉文档查看附件《Three books are thrown into the air, their pages fluttering as they soar over the soccer field, land\_cogvideo.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljgas4zhkz6wzeez&utm_medium=im_card&utm_scene=person_space&utm_source=im) **CogVideoX-5B**  |  [请至钉钉文档查看附件《Three books are thrown into the air, their pages fluttering as they soar over the soccer field, land\_hunyuan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljge3asvsn3sp0o8f&utm_medium=im_card&utm_scene=person_space&utm_source=im) **HunyuanVideo**  |  [请至钉钉文档查看附件《Three books are thrown into the air, their pages fluttering as they soar over the soccer field, land\_mochi.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljgh1llvcmmxhi1&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Mochi 1**  |
| --- | --- | --- |
|  [请至钉钉文档查看附件《Three books are thrown into the air, their pages fluttering as they soar over the soccer field, land\_opensora\_plan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljgjx6l4cv9onm9fg&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-Plan-v1.3.0**  |  [请至钉钉文档查看附件《Three books are thrown into the air, their pages fluttering as they soar over the soccer field, land\_opensora.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljgnyimntum4dflf8&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-v1.2**  |  [请至钉钉文档查看附件《Three books are thrown into the air, their pages fluttering as they soar over the soccer field, land.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7my0iouigmwn4oe14&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Wan2.1**  |

**Prompt:** Four flickering candles cast shadows as they burn steadily on the balcony, their flames dancing with the gentle breeze.

|  [请至钉钉文档查看附件《Four flickering candles cast shadows as they burn steadily on the balcony, their flames dancing with\_cogvideo.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljhgsqwszlcqp373c&utm_medium=im_card&utm_scene=person_space&utm_source=im) **CogVideoX-5B**  |  [请至钉钉文档查看附件《Four flickering candles cast shadows as they burn steadily on the balcony, their flames dancing with\_hunyuan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljhzb783bc1l4y4fo&utm_medium=im_card&utm_scene=person_space&utm_source=im) **HunyuanVideo**  |  [请至钉钉文档查看附件《Four flickering candles cast shadows as they burn steadily on the balcony, their flames dancing with\_mochi.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lji2ejxv2q8ed8eg9&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Mochi 1**  |
| --- | --- | --- |
|  [请至钉钉文档查看附件《Four flickering candles cast shadows as they burn steadily on the balcony, their flames dancing with\_opensora\_plan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lji5toghw3ae5vqob&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-Plan-v1.3.0**  |  [请至钉钉文档查看附件《Four flickering candles cast shadows as they burn steadily on the balcony, their flames dancing with\_opensora.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lji91fexbx81ijuwp&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-v1.2**  |  [请至钉钉文档查看附件《Four flickering candles cast shadows as they burn steadily on the balcony, their flames dancing with.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7my2ya595uty2qp67q&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Wan2.1**  |

**Prompt:** Two penguins waddle along the beach, occasionally stopping to preen their feathers before continuing their journey across the ocean shore.

|  [请至钉钉文档查看附件《Two penguins waddle along the beach, occasionally stopping to preen their feathers before continuing\_cogvideo.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljklmq4ewphi1oski&utm_medium=im_card&utm_scene=person_space&utm_source=im) **CogVideoX-5B**  |  [请至钉钉文档查看附件《Two penguins waddle along the beach, occasionally stopping to preen their feathers before continuing\_hunyuan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljkpu922fhi6pmbv4&utm_medium=im_card&utm_scene=person_space&utm_source=im) **HunyuanVideo**  |  [请至钉钉文档查看附件《Two penguins waddle along the beach, occasionally stopping to preen their feathers before continuing\_mochi.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljl5df2sj26jfc47i&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Mochi 1**  |
| --- | --- | --- |
|  [请至钉钉文档查看附件《Two penguins waddle along the beach, occasionally stopping to preen their feathers before continuing\_opensora\_plan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljl89u0a9j82z2269t&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-Plan-v1.3.0**  |  [请至钉钉文档查看附件《Two penguins waddle along the beach, occasionally stopping to preen their feathers before continuing\_opensora.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljlbjkgmvrqvfgwn5&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-v1.2**  |  [请至钉钉文档查看附件《Two penguins waddle along the beach, occasionally stopping to preen their feathers before continuing.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7my3lzr359dlgv49cd&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Wan2.1**  |

**Prompt:** In the bustling street, two kids run towards a small dog, bending down to carefully comb its fur, their hands moving swiftly.

|  [请至钉钉文档查看附件《In the bustling street, two kids run towards a small dog, bending down to carefully comb its fur, th\_cogvideo.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljtj84kb7vlwht3n&utm_medium=im_card&utm_scene=person_space&utm_source=im) **CogVideoX-5B**  |  [请至钉钉文档查看附件《In the bustling street, two kids run towards a small dog, bending down to carefully comb its fur, th\_hunyuan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljtme6jx50s2ry9n&utm_medium=im_card&utm_scene=person_space&utm_source=im) **HunyuanVideo**  |  [请至钉钉文档查看附件《In the bustling street, two kids run towards a small dog, bending down to carefully comb its fur, th\_mochi.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljtplc6n7zff50hz8&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Mochi 1**  |
| --- | --- | --- |
|  [请至钉钉文档查看附件《In the bustling street, two kids run towards a small dog, bending down to carefully comb its fur, th\_opensora\_plan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljts0yj1612d7sd1h&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-Plan-v1.3.0**  |  [请至钉钉文档查看附件《In the bustling street, two kids run towards a small dog, bending down to carefully comb its fur, th\_opensora.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7ljtutc58dmvibg5ia&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-v1.2**  |  [请至钉钉文档查看附件《In the bustling street, two kids run towards a small dog, bending down to carefully comb its fur, th.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7my4c37mpp5khlwk8&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Wan2.1**  |

**Prompt:** In the garage, a young girl twirls gracefully, her arms outstretched, perfectly matching the lively country line dance beat.

|  [请至钉钉文档查看附件《In the garage, a young girl twirls gracefully, her arms outstretched, perfectly matching the lively \_cogvideo.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lob3hbtvruv1gs6z&utm_medium=im_card&utm_scene=person_space&utm_source=im) **CogVideoX-5B**  |  [请至钉钉文档查看附件《In the garage, a young girl twirls gracefully, her arms outstretched, perfectly matching the lively \_hunyuan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lob90lj5tc8wg8dch&utm_medium=im_card&utm_scene=person_space&utm_source=im) **HunyuanVideo**  |  [请至钉钉文档查看附件《In the garage, a young girl twirls gracefully, her arms outstretched, perfectly matching the lively \_mochi.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lobdqmcf2yxu35dyc&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Mochi 1**  |
| --- | --- | --- |
|  [请至钉钉文档查看附件《In the garage, a young girl twirls gracefully, her arms outstretched, perfectly matching the lively \_opensora\_plan.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lobhphgpoyao9w52&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-Plan-v1.3.0**  |  [请至钉钉文档查看附件《In the garage, a young girl twirls gracefully, her arms outstretched, perfectly matching the lively \_opensora.mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7lobnktgah083xfwz4&utm_medium=im_card&utm_scene=person_space&utm_source=im) **OpenSora-v1.2**  |  [请至钉钉文档查看附件《In the garage, a young girl twirls gracefully, her arms outstretched, perfectly matching the lively .mp4》](https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCEBn05lR8akx1Z5N?cid=64177395982&corpId=dingd8e1123006514592&iframeQuery=anchorId%3DX02m7log71o0syqws35e47p&utm_medium=im_card&utm_scene=person_space&utm_source=im) **Wan2.1**  |

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
pip install -r requirements.txt

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
```

## Download checkpoints

# 🔧Usage

Firstly sample videos, 

## Sample Videos

*   [ ] Please follow our `sample_demo.py`to create videos. 
    

## Evaluation on the VMBench

### Running the Whole Pipeline

`bash eval.sh`

### Running a Single Metric

# ❤️Acknowledgement

# 📜License

# ✏️Citation