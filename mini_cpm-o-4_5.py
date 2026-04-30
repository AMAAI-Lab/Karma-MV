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
JSON_ROOT = Path("/root/research_backup/causal_qa_apr16_II_clean")
OUTPUT_DIR = Path("/root/research_backup/minicpm_atomic_ISMIR_8pm")
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
    """
    Generates a clip with a silent audio track. 
    Crucial: Do NOT use -an (stripping audio) as it crashes the vLLM server.
    """
    muted_path = TEMP_MUTE_DIR / f"muted_{original_path.name}"
    if not muted_path.exists():
        # -f lavfi -i anullsrc creates a silent audio source
        # -shortest ensures the audio matches the video length
        cmd = [
            "ffmpeg", "-y", 
            "-i", str(original_path),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf", "scale=224:224,fps=1", # Keep your downsampling
            "-c:v", "libx264",
            "-c:a", "aac",                 # Encode the new silent audio
            "-shortest",                   # Match shorter stream (the video)
            "-pix_fmt", "yuv420p",
            str(muted_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return muted_path

def extract_letter(text):
    if not text: return "n/a"
    # Extracts the last mentioned letter in the reasoning block
    matches = re.findall(r'\b([a-d])\b', text.lower())
    return matches[-1] if matches else "n/a"

async def check_server_health(session):
    try:
        async with session.get("http://localhost:8000/v1/models", timeout=5) as resp:
            return resp.status == 200
    except: return False

async def call_omni_atomic(session, prompt, v1, v2, v2_audio):
    """Strictly one call at a time to prevent server disconnects."""
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
                elif resp.status == 500:
                    print(f"      [RETRY] Server Error 500. Engine may be resetting.")
                await asyncio.sleep(10)
        except Exception as e:
            print(f"      [RETRY] Connection lost: {e}. Retrying...")
            await asyncio.sleep(15)
    return ""

async def process_transition(session, semaphore, entry, video_id_lower, video_output_subdir):
    async with semaphore:
        past_name, curr_name = entry.get('past_scene', '').lower(), entry.get('current_scene', '').lower()
        vid_data = VIDEO_MAP.get(video_id_lower)
        if not vid_data: return
        
        p1, p2 = vid_data['files'].get(past_name), vid_data['files'].get(curr_name)
        if not p1 or not p2: return

        trans_id = f"{p1.stem}_to_{p2.stem}"
        save_path = video_output_subdir / f"{trans_id}.json"
        if save_path.exists(): return

        questions = entry.get('questions', [])
        final_results = []

        # --- SEQUENTIAL ATOMIC LOOP ---
        # No asyncio.gather here. We do one call at a time.
        for i, q in enumerate(questions):
            opts_str = " ".join([f"{k}) {v}" for k, v in q['options'].items()])
            
            # Logic for Q3 Masking (i=2)
            v2_audio = p2
            q_prompt = f"Identify the music-video causal link.\nQ: {q['question']}\nOptions: {opts_str}\nAnswer with only the letter."
            
            if i == 2: # Prediction task
                v2_audio = get_muted_video(p2)
                # Your requested Mask Message for Q3
                q_prompt = f"[AUDIO MASKED FOR CLIP 2 - RELY ON VISUALS] {q_prompt}"

            content = await call_omni_atomic(session, q_prompt, p1, p2, v2_audio)
            
            res_item = q.copy()
            res_item['ground_truth'] = res_item.pop('answer', 'n/a')
            res_item['model_answer'] = extract_letter(content)
            res_item['audio_masked'] = (i == 2)
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
            
            with open(JSON_ROOT / qa_filename, 'r') as f:
                data_list = json.load(f)

            print(f"[{idx}/{len(qa_files)}] Processing: {v_id}")
            semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
            for entry in data_list:
                await process_transition(session, semaphore, entry, v_id, subdir)

    print(f"\nFinal Runtime: {(time.perf_counter() - start_total)/60:.2f}m")

if __name__ == "__main__":
    asyncio.run(run_evaluation())