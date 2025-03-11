# Video Motion Benchmark
创建仓库

## Installation
```bash
# 安装Q-Align
cd Q-Align
pip install -e .
# 安装Grounded-Segment-Anything
cd Grounded-Segment-Anything
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install -r requirements.txt
# 安装Groudned-SAM-2
cd Grounded-SAM-2
pip install -e .
```