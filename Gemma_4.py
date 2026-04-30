import json
import os
import requests
import re
from concurrent.futures import ThreadPoolExecutor

# --- PATH CONFIGURATION ---
QA_DIR = "/root/research_backup/causal_qa_apr16_II_clean"           # Folder containing all _qa.json files
FEATURE_DIR = "/root/Causal_Scenes_Features" # Folder containing all _causal_features.json files
OUTPUT_DIR = "/root/research_backup/Gemma_4"     # Where results will be saved
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "google/gemma-4-31B-it"
MAX_WORKERS = 24 

def get_scene_number(scene_name):
    """Extracts the integer from 'scene_040.mp4' or 'scene_040'."""
    match = re.search(r'scene_(\d+)', str(scene_name))
    return int(match.group(1)) if match else -1

def build_prompt(features_data, transition, q_obj, is_predictive=False):
    """Constructs prompt using only features. Excludes answer/explanation."""
    past_id = transition['past_scene'].replace('.mp4', '')
    curr_id = transition['current_scene'].replace('.mp4', '')
    
    f_past = features_data['Scenes'][past_id]
    f_curr = features_data['Scenes'][curr_id]

    prompt = "Analyze the video transition data below and answer the question.\n\n"
    
    # --- PAST SCENE: Full Data ---
    prompt += f"### PAST SCENE DATA ({past_id})\n"
    prompt += f"- Visuals: Brightness {f_past['Brightness']}, Intensity {f_past['Total_Visual_Intensity']}, Motion {f_past['Motion']}, Saturation {f_past['Saturation']}\n"
    prompt += f"- Audio: Loudness {f_past['Average Loudness (dB)']} dB, Key {f_past['Key']}, Moods: {', '.join([m['label'] for m in f_past['Moods']])}\n"
    prompt += f"- Content: Objects: {f_past['Detected_Objects']}, Action: {f_past.get('Action', 'N/A')}\n\n"

    # --- CURRENT SCENE: Logic for Q3 Masking ---
    prompt += f"### CURRENT SCENE DATA ({curr_id})\n"
    prompt += f"- Visuals: Brightness {f_curr['Brightness']}, Intensity {f_curr['Total_Visual_Intensity']}, Motion {f_curr['Motion']}, Saturation {f_curr['Saturation']}\n"
    prompt += f"- Content: Objects: {f_curr['Detected_Objects']}, Action: {f_curr.get('Action', 'N/A')}\n"

    if is_predictive:
        # Mask Audio for Question 3 only
        prompt += "- Audio: [DATA MASKED FOR PREDICTION VALIDATION]\n\n"
        prompt += "### TASK\nPredict the audio outcome based on the visual shifts and past state.\n"
    else:
        # Full Audio for Questions 1, 2, and 4
        prompt += f"- Audio: Loudness {f_curr['Average Loudness (dB)']} dB, Key {f_curr['Key']}, Moods: {', '.join([m['label'] for m in f_curr['Moods']])}\n\n"

    # --- QUESTION BLOCK ---
    prompt += f"### QUESTION\n{q_obj['question']}\nOptions:\n"
    for k, v in q_obj['options'].items():
        prompt += f"{k}) {v}\n"
    
    prompt += "\nINSTRUCTION: Output only the letter (a, b, c, or d)."
    return prompt

def get_inference(prompt):
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 5}
    try:
        res = requests.post(API_URL, json=payload, timeout=60).json()
        ans = res['choices'][0]['message']['content'].strip().lower()
        match = re.search(r'[a-d]', ans)
        return match.group(0) if match else "n/a"
    except:
        return "error"

def process_video(qa_filename):
    video_id = qa_filename.replace('_qa.json', '')
    feat_filename = f"{video_id}_causal_features.json"
    
    # Load files
    with open(os.path.join(QA_DIR, qa_filename)) as f1, open(os.path.join(FEATURE_DIR, feat_filename)) as f2:
        qa_data, feat_data = json.load(f1), json.load(f2)

    for transition in qa_data:
        p_num = get_scene_number(transition['past_scene'])
        c_num = get_scene_number(transition['current_scene'])

        # FILTER: Only consecutive pairs (N and N+1)
        if p_num != -1 and c_num == p_num + 1:
            trans_name = f"{transition['past_scene'].replace('.mp4', '')}_to_{transition['current_scene'].replace('.mp4', '')}"
            
            output_obj = {
                "video_id": video_id,
                "transition": trans_name,
                "thought_process": "",
                "results": []
            }

            for i, q in enumerate(transition['questions']):
                # Mask Audio for Q3 (index 2)
                is_q3 = (i == 2)
                prompt = build_prompt(feat_data, transition, q, is_predictive=is_q3)
                model_answer = get_inference(prompt)

                output_obj["results"].append({
                    "question": q['question'],
                    "options": q['options'],
                    "explanation": q.get('explanation', ""),
                    "ground_truth_answer": q.get('answer', ""),
                    "model_answer": model_answer
                })

            # Create specific Video Folder in OutputDir
            save_path = os.path.join(OUTPUT_DIR, video_id)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f"{trans_name}.json"), "w") as f:
                json.dump(output_obj, f, indent=4)

def run_all():
    qa_files = [f for f in os.listdir(QA_DIR) if f.endswith('_qa.json')]
    print(f"Starting inference for {len(qa_files)} videos...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_video, qa_files)

if __name__ == "__main__":
    run_all()