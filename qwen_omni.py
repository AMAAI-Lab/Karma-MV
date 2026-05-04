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
OUTPUT_DIR = Path("/root/research_backup/qwen_omni4May")
TEMP_MUTE_DIR = VIDEO_ROOT / "muted_scenes_temp"
SERVER_URL = "http://localhost:8000/v1/chat/completions"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_MUTE_DIR, exist_ok=True)

# 4 GPUs: Start with lower concurrency to ensure stability
CONCURRENT_LIMIT = 2 

# Pre-indexing logic for path robustness
VIDEO_MAP = {}
if VIDEO_ROOT.exists():
    for vid_dir in VIDEO_ROOT.iterdir():
        if vid_dir.is_dir():
            v_id_lower = vid_dir.name.lower()
            reg = {f.name.lower(): f for f in vid_dir.glob("*.mp4")}
            reg.update({f.stem.lower(): f for f in vid_dir.glob("*.mp4")})
            VIDEO_MAP[v_id_lower] = {"files": reg}

def optimize_video_for_llm(original_path, mute=False):
    """
    Compresses video to 2 FPS and 360p to drastically reduce LLM token count.
    Saves the optimized files in the temporary directory.
    """
    suffix = "_muted" if mute else "_audio"
    opt_path = TEMP_MUTE_DIR / f"opt_{original_path.stem}{suffix}.mp4"
    
    if not opt_path.exists():
        # -vf "fps=2,scale=-2:360": 2 frames per second, 360p height (width auto-scales)
        cmd = [
            "ffmpeg", "-y", "-i", str(original_path), 
            "-vf", "fps=2,scale=-2:360"
        ]
        
        if mute:
            cmd.append("-an") # Remove audio
            
        cmd.append(str(opt_path))
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    return opt_path

def get_muted_video(original_path):
    """Generates a muted copy of the video for the prediction task."""
    muted_path = TEMP_MUTE_DIR / f"muted_{original_path.name}"
    if not muted_path.exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(original_path), 
            "-an", "-c", "copy", str(muted_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return muted_path

def extract_letters(text, expected_count=None):
    """Robustly extracts a, b, c, or d from model responses."""
    if not text: return []
    
    # Matches formats like: "A", "a)", "(b)", "**C**", "Q1: d", "Option A", "1. a"
    # (?i) makes it case-insensitive
    pattern = r'(?i)(?:^|[\s\n]|q\d\s*[:.)-]\s*|option\s+)[*\[(]*([a-d])[*)\]]*(?:[.:,\s\n]|$)'
    matches = re.findall(pattern, text)
    
    # Clean up and lowercase
    results = [m.lower() for m in matches if m]
    
    # Debugging: If we didn't find enough letters, print the raw text so we can see why
    if expected_count is not None and len(results) < expected_count:
        print(f"\n--- PARSE WARNING ---")
        print(f"Expected {expected_count} answers, but extracted {len(results)}.")
        print(f"Raw Model Output:\n{text}")
        print(f"Extracted: {results}")
        print(f"---------------------\n")
        
    return results

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
        past_scene_data = entry.get('past_scene', {})
        curr_scene_data = entry.get('current_scene', {})
        
        past_name = past_scene_data.get('name', '').strip() if isinstance(past_scene_data, dict) else str(past_scene_data).strip()
        curr_name = curr_scene_data.get('name', '').strip() if isinstance(curr_scene_data, dict) else str(curr_scene_data).strip()
        
        vid_data = VIDEO_MAP.get(video_id_lower)
        if not vid_data: return None
        
        path_past = vid_data['files'].get(past_name.lower())
        path_curr = vid_data['files'].get(curr_name.lower())
        if not path_past or not path_curr: return None

        trans_id = f"{path_past.stem}_to_{path_curr.stem}"
        save_path = video_output_subdir / f"{trans_id}.json"
        if save_path.exists(): return 0.0

        questions = entry.get('questions', [])
        
        full_audio_indices = []
        muted_audio_indices = []
        
        for i, q in enumerate(questions):
            if q.get('type') == 'Predictive':
                muted_audio_indices.append(i)
            else:
                full_audio_indices.append(i)
        
        tasks = []
        
        # Build Phase 1 Prompt (Full Audio)
# Optimize the videos to prevent token limit crashes (Context Window overflow)
        path_past_opt = optimize_video_for_llm(path_past, mute=False)
        path_curr_opt = optimize_video_for_llm(path_curr, mute=False)

        # Build Phase 1 Prompt (Full Audio)
        if full_audio_indices:
            prompt_full = "Analyze these clips. Provide ONLY the correct letter option (a, b, c, or d) for the following questions:\n\n"
            for i in full_audio_indices:
                q = questions[i]
                opts = " ".join([f"{k}) {v}" for k, v in q['options'].items()])
                prompt_full += f"Q{i+1}: {q['question']}\nOptions: {opts}\n\n"
            
            # Send the OPTIMIZED videos instead of the massive original files
            tasks.append(call_omni(session, prompt_full, path_past_opt, path_curr_opt))
        else:
            tasks.append(asyncio.sleep(0))
            
        # Build Phase 2 Prompt (Muted Audio for Predictive Tasks)
        if muted_audio_indices:
            # Generate an optimized, muted version of the current video
            path_curr_muted_opt = optimize_video_for_llm(path_curr, mute=True)
            prompt_muted = "PREDICTION TASK: Provide ONLY the correct letter option (a, b, c, or d) for the following questions:\n\n"
            for i in muted_audio_indices:
                q = questions[i]
                opts = " ".join([f"{k}) {v}" for k, v in q['options'].items()])
                prompt_muted += f"Q{i+1}: {q['question']}\nOptions: {opts}\n\n"
                
            # Send the OPTIMIZED videos
            tasks.append(call_omni(session, prompt_muted, path_past_opt, path_curr_muted_opt))
        else:
            tasks.append(asyncio.sleep(0))

        # Build Phase 2 Prompt (Muted Audio for Predictive Tasks)
        if muted_audio_indices:
            path_curr_muted = get_muted_video(path_curr)
            prompt_muted = "PREDICTION TASK: Provide ONLY the correct letter option (a, b, c, or d) for the following questions:\n\n"
            for i in muted_audio_indices:
                q = questions[i]
                opts = " ".join([f"{k}) {v}" for k, v in q['options'].items()])
                prompt_muted += f"Q{i+1}: {q['question']}\nOptions: {opts}\n\n"
            tasks.append(call_omni(session, prompt_muted, path_past, path_curr_muted))
        else:
            tasks.append(asyncio.sleep(0))

        # Parallel Execution
        resps = await asyncio.gather(*tasks)
        
        # Passing expected counts so we can print warnings if the LLM output is malformed
        letters_full = extract_letters(resps[0] if resps[0] else "", expected_count=len(full_audio_indices))
        letters_muted = extract_letters(resps[1] if resps[1] else "", expected_count=len(muted_audio_indices))

        # Consolidation
        final_results = []
        full_idx = 0
        muted_idx = 0
        
        for i in range(len(questions)):
            q = questions[i].copy()
            q['ground_truth_answer'] = q.pop('answer', 'n/a')
            
            if i in muted_audio_indices:
                q['model_answer'] = letters_muted[muted_idx] if muted_idx < len(letters_muted) else "n/a"
                q['audio_masked'] = True
                muted_idx += 1
            else:
                q['model_answer'] = letters_full[full_idx] if full_idx < len(letters_full) else "n/a"
                q['audio_masked'] = False
                full_idx += 1
                
            final_results.append(q)

        with open(save_path, 'w') as f:
            json.dump({"video_id": video_id_lower, "results": final_results}, f, indent=4)
        return True

async def run_evaluation():
    start_total = time.perf_counter()
    
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
            file_path = JSON_ROOT / qa_filename
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                decoder = json.JSONDecoder()
                parsed_objs = []
                pos = 0
                
                while pos < len(content):
                    content_str = content[pos:].lstrip()
                    if not content_str:
                        break
                    obj, read_index = decoder.raw_decode(content_str)
                    parsed_objs.append(obj)
                    pos += read_index + (len(content[pos:]) - len(content_str))

                video_id_from_json = None
                data_list = []
                
                for obj in parsed_objs:
                    if isinstance(obj, dict) and "Video_id" in obj:
                        video_id_from_json = obj["Video_id"]
                    elif isinstance(obj, list):
                        data_list.extend(obj)
                    elif isinstance(obj, dict):
                        for val in obj.values():
                            if isinstance(val, list):
                                data_list.extend(val)

                if video_id_from_json:
                    video_id = str(video_id_from_json).strip()
                else:
                    video_id = qa_filename.replace('_qa.json', '').strip()
                    
                video_id_lower = video_id.lower()
                
            except Exception as e:
                print(f"Error loading and parsing {qa_filename}: {e}")
                continue

            if not data_list:
                print(f"[{idx}/{len(qa_files)}] Skipping {video_id}: No valid question data found.")
                continue
                
            if video_id_lower not in VIDEO_MAP:
                print(f"[{idx}/{len(qa_files)}] Skipping {video_id}: Folder not indexed.")
                continue
                
            video_output_subdir = OUTPUT_DIR / video_id
            video_output_subdir.mkdir(parents=True, exist_ok=True)

            print(f"[{idx}/{len(qa_files)}] Processing: {video_id} ({len(data_list)} pairs)")
            tasks = [process_transition(session, semaphore, entry, video_id_lower, video_output_subdir) 
                     for entry in data_list]
            await asyncio.gather(*tasks)

    print(f"\nFinal Runtime: {(time.perf_counter() - start_total)/60:.2f}m")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
