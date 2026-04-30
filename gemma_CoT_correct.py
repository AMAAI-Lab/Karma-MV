import json
import os
import requests
import re
import pandas as pd
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
QA_DIR = "/root/research_backup/causal_qa_apr16_II_clean"
FEATURE_DIR = "/root/Causal_Scenes_Features"
OUTPUT_DIR = "/root/research_backup/Gemma_4_ISMIR_Final"
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "google/gemma-4-31B-it"
MAX_WORKERS = 32 

# --- KG INITIALIZATION ---
print("Initializing Knowledge Graph...", flush=True)
G = nx.DiGraph()
entities_df = pd.read_parquet("entities.parquet")
relations_df = pd.read_parquet("relationships.parquet")
for _, row in entities_df.iterrows():
    G.add_node(row['id'], title=str(row.get('title', row['id'])).lower())
for _, row in relations_df.iterrows():
    G.add_edge(row['source'], row['target'], description=row['description'])

# --- ADVANCED KG PATH RETRIEVAL ---

def get_causal_path_facts(question_text, feature_names):
    """Finds 2-hop causal paths to bridge visual causes to audio effects."""
    path_facts = []
    keywords = set(re.findall(r'\b\w{4,}\b', question_text.lower()))
    
    start_nodes = [n for n, d in G.nodes(data=True) if any(f.lower() in d['title'] for f in feature_names)]
    target_nodes = [n for n, d in G.nodes(data=True) if any(kw in d['title'] for kw in keywords)]

    for start in start_nodes[:5]:
        for target in target_nodes[:5]:
            try:
                path = nx.shortest_path(G, start, target, weight=None)
                if len(path) <= 3: 
                    for i in range(len(path)-1):
                        u, v = path[i], path[i+1]
                        desc = G[u][v]['description']
                        path_facts.append(f"LOGIC: [{u}] -> [{v}] ({desc})")
            except nx.NetworkXNoPath:
                continue
            except Exception:
                continue
    
    return "\n".join(list(set(path_facts))[:3])

# --- PROMPT REFINEMENT ---

def get_safe_action(feature_dict):
    """Safely extracts the action label to prevent IndexError."""
    action_list = feature_dict.get('Action', [])
    if action_list and isinstance(action_list, list) and len(action_list) > 0:
        return action_list[0].get('label', 'N/A')
    return 'N/A'

def build_refined_prompt(f_p, f_c, q_obj, idx):
    is_q3, is_q4 = (idx == 2), (idx == 3)
    
    deltas = []
    for f in ['Brightness', 'Total_Visual_Intensity', 'Motion', 'Contrast']:
        p, c = float(f_p.get(f, 0)), float(f_c.get(f, 0))
        trend = "INCREASED" if c > p else "DECREASED"
        deltas.append(f"- {f}: {trend} (Current: {c:.2f} vs Past: {p:.2f})")
    
    # Safely get actions
    p_act = get_safe_action(f_p)
    c_act = get_safe_action(f_c)
    action_msg = f"- Action Shift: from '{p_act}' to '{c_act}'"
    
    kg_facts = get_causal_path_facts(q_obj['question'], ['Intensity', 'Brightness', 'Motion'])

    prompt = (
        "### ROLE: Causal Multimedia Expert\n"
        "Analyze the transition logic. Use the Knowledge Graph (KG) as a structural constraint.\n\n"
        "### KG CAUSAL LOGIC (Rules of the World)\n"
        f"{kg_facts if kg_facts else 'Use standard audio-visual causal principles.'}\n\n"
        "### REALITY: ACTUAL TRANSITION OBSERVATIONS\n"
        f"{action_msg}\n" + "\n".join(deltas) + "\n"
    )

    if is_q4:
        prompt += (
            "\n### TASK: COUNTERFACTUAL REASONING\n"
            "The question asks about a HYPOTHESIS that differs from the reality above.\n"
            "1. Contrast the HYPOTHESIS with the ACTUAL observation.\n"
            "2. Apply the KG LOGIC to the hypothetical change to find the pivot.\n"
        )
    elif is_q3:
        prompt += "\n### TASK: PREDICTIVE INFERENCE\nAudio is MASKED. Use the Action Shift and Visual Intensity to predict the Audio.\n"

    prompt += f"\n### QUESTION\n{q_obj['question']}\n\nOptions:\n"
    for k, v in q_obj['options'].items():
        prompt += f"{k}) {v}\n"
    
    prompt += "\nRespond ONLY in JSON format: {'reasoning': '...', 'answer': '...'}"
    return prompt

# --- INFERENCE ---

def get_inference(prompt):
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    try:
        res = requests.post(API_URL, json=payload, timeout=90).json()
        raw = res['choices'][0]['message']['content'].strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get('reasoning', ''), data.get('answer', 'n/a').lower()
        return "Parse Error", "error"
    except Exception as e: 
        return f"Network/API Error: {str(e)}", "error"

def get_scene_number(scene_name):
    """Safely extracts the integer from the scene name."""
    match = re.search(r'scene_(\d+)', str(scene_name))
    return int(match.group(1)) if match else -1

def process_video(qa_filename):
    video_id = qa_filename.replace('_qa.json', '')
    feat_path = os.path.join(FEATURE_DIR, f"{video_id}_causal_features.json")
    
    # Explicitly catch and print file loading errors
    try:
        with open(os.path.join(QA_DIR, qa_filename)) as f1, open(feat_path) as f2:
            qa_data = json.load(f1)
            feat_data = json.load(f2)
    except Exception as e:
        print(f"[FILE ERROR] Skipping {video_id}: {str(e)}", flush=True)
        return

    # Wrap the entire processing logic to catch dictionary/index crashes
    try:
        for transition in qa_data:
            p_id = transition['past_scene'].replace('.mp4', '')
            c_id = transition['current_scene'].replace('.mp4', '')
            
            p_num = get_scene_number(p_id)
            c_num = get_scene_number(c_id)
            
            # Filter for consecutive scenes safely
            if p_num != -1 and c_num == p_num + 1:
                
                # Check if scenes actually exist in features
                if p_id not in feat_data['Scenes'] or c_id not in feat_data['Scenes']:
                    print(f"[DATA WARNING] Missing scenes in feature file for {video_id} ({p_id} -> {c_id})", flush=True)
                    continue

                results = []
                for i, q in enumerate(transition['questions']):
                    prompt = build_refined_prompt(feat_data['Scenes'][p_id], feat_data['Scenes'][c_id], q, i)
                    reasoning, ans = get_inference(prompt)
                    
                    final_ans = re.search(r'[a-d]', str(ans))
                    final_ans = final_ans.group(0) if final_ans else "n/a"

                    print(f"[V2 RUN] {video_id} Q{i+1}: {final_ans}", flush=True)

                    results.append({
                        "question": q['question'],
                        "model_reasoning": reasoning,
                        "model_answer": final_ans,
                        "ground_truth_answer": q.get('answer', "")
                    })

                out_dir = os.path.join(OUTPUT_DIR, video_id)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{p_id}_to_{c_id}.json"), "w") as f:
                    json.dump({"results": results}, f, indent=4)

    except Exception as e:
        print(f"[PROCESSING ERROR] Thread crashed on {video_id}: {str(e)}", flush=True)

def run():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    qa_files = [f for f in os.listdir(QA_DIR) if f.endswith('_qa.json')]
    print(f"Starting V2 Accuracy Optimization for {len(qa_files)} videos...", flush=True)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Using executor.submit instead of map so we can catch thread-level exceptions
        futures = {executor.submit(process_video, qa_file): qa_file for qa_file in qa_files}
        
        for future in as_completed(futures):
            qa_file = futures[future]
            try:
                future.result() # This forces any unhandled exception in the thread to be raised here
            except Exception as e:
                print(f"[FATAL THREAD ERROR] File {qa_file} caused a catastrophic crash: {e}", flush=True)

if __name__ == "__main__":
    run()