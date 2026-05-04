import os
import json
import asyncio
import aiohttp
import time
import re
import subprocess
from pathlib import Path

# --- CONFIGURATION ---
VIDEO_ROOT = Path("/mnt/data/archishman/Filtered_scenes")
JSON_ROOT = Path("/root/research_backup/hf_ready_dataset")
OUTPUT_DIR = Path("/root/research_backup/minicpm_May4")
TEMP_MUTE_DIR = VIDEO_ROOT / "muted_scenes_temp"
SERVER_URL = "http://localhost:8000/v1/chat/completions"

# 1 transition at a time, 1 question at a time
CONCURRENT_LIMIT = 4 
MAX_RETRIES = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_MUTE_DIR, exist_ok=True)

# 1. ROBUST PRE-INDEXING
VIDEO_MAP = {}
if VIDEO_ROOT.exists():
    for vid_dir in VIDEO_ROOT.iterdir():
        if vid_dir.is_dir():
            v_id_lower = vid_dir.name.lower()
            reg = {f.name.lower(): f for f in vid_dir.glob("*.mp4")}
            reg.update({f.stem.lower(): f for f in vid_dir.glob("*.mp4")})
            VIDEO_MAP[v_id_lower] = {"files": reg}

def get_muted_video(original_path):
    muted_path = TEMP_MUTE_DIR / f"muted_{original_path.name}"
    if not muted_path.exists():
        cmd = [
            "ffmpeg", "-y", 
            "-i", str(original_path),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf", "scale=224:224,fps=1", 
            "-c:v", "libx264",
            "-c:a", "aac",                 
            "-shortest",                   
            "-pix_fmt", "yuv420p",
            str(muted_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return muted_path

def extract_letter(text):
    if not text: return "n/a"
    matches = re.findall(r'\b([a-d])\b', text.lower())
    return matches[-1] if matches else "n/a"

async def call_omni_atomic(session, prompt, v1, v2, v2_audio):
    payload = {
        "model": "openbmb/MiniCPM-o-4_5",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"file://{v1.absolute()}"}},
            {"type": "audio_url", "audio_url": {"url": f"file://{v1.absolute()}"}},
            {"type": "video_url", "video_url": {"url": f"file://{v2.absolute()}"}},
            {"type": "audio_url", "audio_url": {"url": f"file://{v2_audio.absolute()}"}}
        ]}],
        "temperature": 0,
        "max_tokens": 1024
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(SERVER_URL, json=payload, timeout=600) as resp:
                if resp.status == 200:
                    res_json = await resp.json()
                    return res_json['choices'][0]['message'].get('content', '')
                await asyncio.sleep(10)
        except Exception as e:
            await asyncio.sleep(15)
    return ""

async def process_transition(session, semaphore, entry, video_id_lower, video_output_subdir):
    async with semaphore:
        # Handling the nested dictionary structure[cite: 1]
        past_scene_info = entry.get('past_scene', {})
        curr_scene_info = entry.get('current_scene', {})
        
        past_name = past_scene_info.get('name', '').lower()
        curr_name = curr_scene_info.get('name', '').lower()
        
        vid_data = VIDEO_MAP.get(video_id_lower)
        if not vid_data: return
        
        p1 = vid_data['files'].get(past_name) or vid_data['files'].get(Path(past_name).stem)
        p2 = vid_data['files'].get(curr_name) or vid_data['files'].get(Path(curr_name).stem)
        
        if not p1 or not p2: return

        trans_id = f"{p1.stem}_to_{p2.stem}"
        save_path = video_output_subdir / f"{trans_id}.json"
        if save_path.exists(): return

        questions = entry.get('questions', [])
        final_results = []

        for i, q in enumerate(questions):
            opts_str = " ".join([f"{k}) {v}" for k, v in q['options'].items()])
            v2_audio = p2
            q_prompt = f"Identify the music-video causal link.\nQ: {q['question']}\nOptions: {opts_str}\nAnswer with only the letter."
            
            if i == 1: # Prediction Masking
                v2_audio = get_muted_video(p2)
                q_prompt = f"[AUDIO MASKED FOR CLIP 2 - RELY ON VISUALS] {q_prompt}"

            content = await call_omni_atomic(session, q_prompt, p1, p2, v2_audio)
            
            res_item = q.copy()
            res_item['ground_truth'] = res_item.pop('answer', 'n/a') # Correct mapping[cite: 1]
            res_item['model_answer'] = extract_letter(content)
            res_item['audio_masked'] = (i == 1)
            final_results.append(res_item)

        with open(save_path, 'w') as f:
            json.dump({"transition": trans_id, "results": final_results}, f, indent=4)
        print(f"      [SUCCESS] {trans_id}")

async def run_evaluation():
    start_total = time.perf_counter()
    qa_files = sorted([f for f in os.listdir(JSON_ROOT) if f.endswith('_qa.json')])
    
    async with aiohttp.ClientSession() as session:
        for idx, qa_filename in enumerate(qa_files, 1):
            v_id = qa_filename.replace('_qa.json', '').lower()
            if v_id not in VIDEO_MAP: continue
                
            subdir = OUTPUT_DIR / v_id
            subdir.mkdir(parents=True, exist_ok=True)
            
            # Robust parsing for files containing multiple JSON objects[cite: 1]
            data_list = []
            with open(JSON_ROOT / qa_filename, 'r') as f:
                content = f.read().strip()
                decoder = json.JSONDecoder()
                pos = 0
                while pos < len(content):
                    try:
                        # Skip leading whitespace
                        while pos < len(content) and content[pos].isspace():
                            pos += 1
                        if pos >= len(content): break
                        
                        obj, next_pos = decoder.raw_decode(content, pos)
                        if isinstance(obj, list):
                            data_list = obj
                            break # Found the main list[cite: 1]
                        pos = next_pos
                    except json.JSONDecodeError:
                        break

            print(f"[{idx}/{len(qa_files)}] Processing: {v_id}")
            semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
            tasks = []
            for entry in data_list:
                if isinstance(entry, dict) and "questions" in entry:
                    tasks.append(process_transition(session, semaphore, entry, v_id, subdir))
            
            if tasks:
                await asyncio.gather(*tasks)

    print(f"\nFinal Runtime: {(time.perf_counter() - start_total)/60:.2f}m")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
