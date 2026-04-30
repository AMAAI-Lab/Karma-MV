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
OUTPUT_DIR = Path("/root/research_backup/omni_25_7B_16Apr")
TEMP_MUTE_DIR = VIDEO_ROOT / "muted_scenes_temp"
SERVER_URL = "http://localhost:8000/v1/chat/completions"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_MUTE_DIR, exist_ok=True)

# 4 GPUs: Start with lower concurrency to ensure stability
CONCURRENT_LIMIT = 8 

# Pre-indexing logic for path robustness
VIDEO_MAP = {}
if VIDEO_ROOT.exists():
    for vid_dir in VIDEO_ROOT.iterdir():
        if vid_dir.is_dir():
            v_id_lower = vid_dir.name.lower()
            reg = {f.name.lower(): f for f in vid_dir.glob("*.mp4")}
            reg.update({f.stem.lower(): f for f in vid_dir.glob("*.mp4")})
            VIDEO_MAP[v_id_lower] = {"files": reg}

def get_muted_video(original_path):
    """Generates a muted copy of the video for the prediction task."""
    muted_path = TEMP_MUTE_DIR / f"muted_{original_path.name}"
    if not muted_path.exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(original_path), 
            "-an", "-c", "copy", str(muted_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return muted_path

def extract_letters(text):
    """Robustly extracts a, b, c, or d from model responses."""
    if not text: return []
    # Matches 'q1: a', '1. a', or just 'a' in a list
    return re.findall(r'(?:q\d|[\d]\.)?[:\s]*([a-d])(?:\s|$|\.)', text.lower())

async def check_server_health(session):
    try:
        async with session.get("http://localhost:8000/v1/models", timeout=5) as resp:
            return resp.status == 200
    except:
        return False    

async def call_omni(session, prompt, v1, v2):
    payload = {
        "model": "Qwen/Qwen2.5-Omni-7B",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"file://{v1.absolute()}"}},
            {"type": "video_url", "video_url": {"url": f"file://{v2.absolute()}"}}
        ]}],
        "temperature": 0,
        "max_tokens": 64
    }
    try:
        async with session.post(SERVER_URL, json=payload, timeout=300) as resp:
            if resp.status == 200:
                res = await resp.json()
                return res['choices'][0]['message']['content']
    except Exception as e:
        print(f"!!! API CALL FAILED: {e}")
    return ""

async def process_transition(session, semaphore, entry, video_id_lower, video_output_subdir):
    async with semaphore:
        past_name = entry.get('past_scene', '').strip()
        curr_name = entry.get('current_scene', '').strip()
        vid_data = VIDEO_MAP.get(video_id_lower)
        
        if not vid_data: return None
        path_past = vid_data['files'].get(past_name.lower())
        path_curr = vid_data['files'].get(curr_name.lower())
        if not path_past or not path_curr: return None

        trans_id = f"{path_past.stem}_to_{path_curr.stem}"
        save_path = video_output_subdir / f"{trans_id}.json"
        if save_path.exists(): return 0.0

        questions = entry.get('questions', [])
        
        # Build Phase 1 Prompt (Full Audio)
        prompt_full = "Analyze these clips. Provide ONLY the correct letter option (a, b, c, or d) for Q1, Q2, and Q4:\n\n"
        for i in [0, 1, 3]:
            q = questions[i]
            opts = " ".join([f"{k}) {v}" for k, v in q['options'].items()])
            prompt_full += f"Q{i+1}: {q['question']}\nOptions: {opts}\n\n"

        # Build Phase 2 Prompt (Muted Audio for Q3)
        path_curr_muted = get_muted_video(path_curr)
        q3 = questions[2]
        opts_q3 = " ".join([f"{k}) {v}" for k, v in q3['options'].items()])
        prompt_q3 = f"PREDICTION TASK: Provide ONLY the correct letter option (a, b, c, or d) for Q3:\n\nQ3: {q3['question']}\nOptions: {opts_q3}"
        
        # Parallel Execution: ONE SET OF CALLS ONLY
        tasks = [
            call_omni(session, prompt_full, path_past, path_curr),
            call_omni(session, prompt_q3, path_past, path_curr_muted)
        ]
        resps = await asyncio.gather(*tasks)
        
        # Use robust extraction consistently
        letters_full = extract_letters(resps[0])
        letter_q3 = extract_letters(resps[1])

        # Consolidation
        final_results = []
        full_idx = 0
        for i in range(len(questions)):
            q = questions[i].copy()
            q['ground_truth_answer'] = q.pop('answer', 'n/a')
            
            if i == 2: # Prediction Task (Q3)
                q['model_answer'] = letter_q3[0] if letter_q3 else "n/a"
                q['audio_masked'] = True
            else: # Description/Counterfactual Tasks (Q1, Q2, Q4)
                q['model_answer'] = letters_full[full_idx] if full_idx < len(letters_full) else "n/a"
                full_idx += 1
            final_results.append(q)

        with open(save_path, 'w') as f:
            json.dump({"video_id": video_id_lower, "results": final_results}, f, indent=4)
        return True

# run_evaluation() logic remains the same...

async def run_evaluation():
    start_total = time.perf_counter()
    
    # Ensure JSON_ROOT exists
    if not JSON_ROOT.exists():
        print(f"!!! ERROR: JSON_ROOT {JSON_ROOT} does not exist.")
        return

    qa_files = sorted([f for f in os.listdir(JSON_ROOT) if f.endswith('_qa.json')])
    
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    async with aiohttp.ClientSession() as session:
        if not await check_server_health(session):
            print("!!! ERROR: vLLM Server not responding at localhost:8000")
            return

        for idx, qa_filename in enumerate(qa_files, 1):
            video_id = qa_filename.replace('_qa.json', '').strip()
            video_id_lower = video_id.lower()
            
            if video_id_lower not in VIDEO_MAP:
                print(f"[{idx}/{len(qa_files)}] Skipping {video_id}: Folder not indexed.")
                continue
                
            video_output_subdir = OUTPUT_DIR / video_id
            video_output_subdir.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(JSON_ROOT / qa_filename, 'r') as f:
                    data_list = json.load(f)
            except Exception as e:
                print(f"Error loading {qa_filename}: {e}")
                continue

            print(f"[{idx}/{len(qa_files)}] Processing: {video_id} ({len(data_list)} pairs)")
            tasks = [process_transition(session, semaphore, entry, video_id_lower, video_output_subdir) 
                     for entry in data_list]
            await asyncio.gather(*tasks)

    print(f"\nFinal Runtime: {(time.perf_counter() - start_total)/60:.2f}m")

if __name__ == "__main__":
    asyncio.run(run_evaluation())