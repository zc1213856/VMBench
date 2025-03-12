import sys
import os
import json
import torch

from datetime import datetime
import argparse


def load_json(path, pri=False):
    with open(path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        if pri:
            print(f'load json file: {path}')
    return datas

def save_json(d, path, pri=False):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        if pri:
            print(f'save to json file: {path}')


parser = argparse.ArgumentParser(description="args")

parser.add_argument("-f", "--prompt_path", type=str, default="./prompts/prompts.json", help="prompts json path")
parser.add_argument("-s", "--save_dir", type=str, default="./eval_results/videos", help="results root")

args = parser.parse_args()
print(args)

prompt_path = args.prompt_path
save_dir = args.save_dir

##################### mkdir

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
videos_dir = os.path.join(save_dir, 'videos')
if not os.path.exists(videos_dir):
    os.makedirs(videos_dir, exist_ok=True)
results_path = os.path.join(save_dir, 'results.json')
#####################


##################### example: cogvideo

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

#####################

def text_to_video(prompt, filepath):
    # input: prompt for video generation
    # 'A green turtle swims alongside a school of dolphins.' 
    #
    # input: path of generated video
    # './results/20250211_115555/videos/A green turtle swims alongside a school of dolphins.mp4'
    try:
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        export_to_video(video, filepath, fps=8)

    except:
        print(f"Error processing prompt: {prompt}")

def video_generation():
    datas = load_json(prompt_path)
    if os.path.exists(results_path):
        results = load_json(results_path)
        print(f'continue from: {results_path}')
    else:
        results = datas.copy()
    r = []
    for i in range(len(datas)):
        item = datas[i].copy()
        prompt = item['prompt']      
        filename = item['index']
        video_path = os.path.join(save_dir, f'{filename}.mp4')
        if os.path.exists(video_path): # skip
            item['filepath'] = video_path
            save_json(r, results_path)
            continue
        text_to_video(item['prompt'], video_path)
        assert os.path.exists(video_path)
        item['filepath'] = video_path
        r.append(item)
        save_json(r, results_path)


def main():
    video_generation()

if __name__=='__main__':
    main()
