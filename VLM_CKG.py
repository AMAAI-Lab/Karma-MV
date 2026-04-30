import os
import json
import asyncio
import aiohttp
import time
import re
import subprocess
import pandas as pd
import networkx as nx
from pathlib import Path

# --- CONFIGURATION ---
VIDEO_ROOT = Path("/mnt/data/archishman/Filtered_scenes")
JSON_ROOT = Path("/root/research_backup/causal_qa_apr16_II_clean")
OUTPUT_DIR = Path("/root/research_backup/omni_graphed_21Apr_NetX")
TEMP_MUTE_DIR = VIDEO_ROOT / "muted_scenes_temp"
SERVER_URL = "http://localhost:8000/v1/chat/completions"

# Knowledge Graph Paths
ENTITIES_PATH = "entities.parquet"
RELATIONS_PATH = "relationships.parquet"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_MUTE_DIR, exist_ok=True)

# Resource Control
CONCURRENT_LIMIT = 4 
MAX_RETRIES = 3

# --- NETWORKX KG INITIALIZATION ---
print("Building NetworkX Graph Store from research parquets...")
entities_df = pd.read_parquet(ENTITIES_PATH)
relations_df = pd.read_parquet(RELATIONS_PATH)

# Build Directed Graph
G = nx.DiGraph()

# Add Nodes with titles for searching
for _, row in entities_df.iterrows():
    G.add_node(row['id'], title=str(row.get('title', row['id'])).lower())

# Add Edges with causal descriptions
for _, row in relations_df.iterrows():
    G.add_edge(row['source'], row['target'], description=row['description'])

# Pre-calculate degree for topological weighting
node_degrees = dict(G.degree())

def get_graph_context_by_query(question_text):
    """
    Retrieves KG facts using NetworkX traversal.
    Fast keyword matching + 1-hop causal expansion.
    """
    candidate_facts = []
    # Remove scene-specific noise and extract keywords
    clean_text = re.sub(r'scene_\d+', '', question_text).lower()
    keywords = set(re.findall(r'\b\w{4,}\b', clean_text))
    
    seen_facts = set()
    transition_keywords = {'change', 'shift', 'increase', 'decrease', 'transition', 'move'}

    # 1. Topological Search
    for node_id, data in G.nodes(data=True):
        node_title = data.get('title', '')
        
        # Keyword match against node title
        if any(kw in node_title for kw in keywords):
            # 2. Causal Traversal (Outgoing Edges)
            for neighbor in G.neighbors(node_id):
                edge_data = G.get_edge_data(node_id, neighbor)
                desc = edge_data.get('description', '')
                
                fact = f"Concept [{node_id}] -> [{neighbor}] via: {desc}"
                if fact not in seen_facts:
                    # Score = Node Centrality * Transition Boost
                    weight = node_degrees.get(node_id, 1)
                    relevance_boost = 1.5 if any(k in desc.lower() for k in transition_keywords) else 1.0
                    
                    candidate_facts.append((fact, weight * relevance_boost))
                    seen_facts.add(fact)
    
    # Sort by centrality (The most "authoritative" concepts first)
    sorted_facts = sorted(candidate_facts, key=lambda x: x[1], reverse=True)
    
    # Return Top 5 to keep VLM focused
    return "\n".join([item[0] for item in sorted_facts[:25]])

# --- HELPERS ---
async def check_server_health(session):
    try:
        async with session.get("http://localhost:8000/v1/models", timeout=5) as resp:
            return resp.status == 200
    except: return False

def get_muted_video(original_path):
    muted_path = TEMP_MUTE_DIR / f"muted_{original_path.name}"
    if not muted_path.exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(original_path), 
            "-an", "-c", "copy", str(muted_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return muted_path

def extract_single_letter(text):
    if not text: return "n/a"
    match = re.search(r'\b([a-d])\b', text.lower())
    return match.group(1) if match else "n/a"

async def call_omni_with_retry(session, prompt, v1, v2):
    payload = {
        "model": "Qwen/Qwen2.5-Omni-7B",
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": f"file://{v1.absolute()}", "fps": 4.0, "max_pixels": 448*448}},
            {"type": "video_url", "video_url": {"url": f"file://{v2.absolute()}", "fps": 4.0, "max_pixels": 448*448}},
            {"type": "text", "text": prompt}
        ]}],
        "temperature": 0, "max_tokens": 16
    }
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(SERVER_URL, json=payload, timeout=300) as resp:
                if resp.status == 200:
                    res = await resp.json()
                    return res['choices'][0]['message']['content']
        except: 
            await asyncio.sleep(2 * (attempt + 1))
    return ""

# Video Indexing
VIDEO_MAP = {}
for vid_dir in VIDEO_ROOT.iterdir():
    if vid_dir.is_dir():
        v_id_lower = vid_dir.name.lower()
        reg = {f.name.lower(): f for f in vid_dir.glob("*.mp4")}
        reg.update({f.stem.lower(): f for f in vid_dir.glob("*.mp4")})
        VIDEO_MAP[v_id_lower] = {"files": reg}

# --- MAIN CORE ---
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
        tasks = []
        question_kg_map = {} 

        print("\n" + "═"*60 + f"\nTransition: {video_id_lower} | {trans_id}\n" + "═"*60)

        for i, q in enumerate(questions):
            opts = " ".join([f"{k}) {v}" for k, v in q['options'].items()])
            
            # Use the new NetworkX retrieval logic
            q_kg_context = get_graph_context_by_query(q['question'])
            question_kg_map[f"Q{i+1}"] = q_kg_context

            fact_count = len(q_kg_context.splitlines()) if q_kg_context else 0
            print(f"  [Q{i+1}] KG Context ({fact_count} facts): {q_kg_context[:80]}...")

            v2_path = path_curr
            mask_msg = ""
            if i == 2: # Q3 logic preserved
                v2_path = get_muted_video(path_curr)
                mask_msg = "PREDICTION TASK: Predict music based on visual changes.\n"

            prompt = (
                f"KNOWLEDGE GRAPH BACKGROUND:\n{q_kg_context if q_kg_context else 'None'}\n\n"
                f"{mask_msg}Analyze the transition clips and provide the answer with ONLY the correct letter option (a, b, c, or d) for the following MCQ. \n\n" 
                f"'KNOWLEDGE GRAPH BACKGROUND' provides a bigger picture on how causality in Music-Video takes place. Can be helpful in REASONING, PREDICTING and COUNTERFACTUAL ANALYSIS (NOTE: Strictly do not use KNOWLEDGE GRAPH BACKGROUND for Normal scene Description questions which ask about simple changes in fetaures).\n\n"
                f"Question: {q['question']}\nOptions: {opts}\n"
                "Answer: [letter]"
            )
            
            tasks.append(call_omni_with_retry(session, prompt, path_past, v2_path))

        raw_responses = await asyncio.gather(*tasks)

        final_results = []
        for idx, raw_text in enumerate(raw_responses):
            q = questions[idx].copy()
            q['ground_truth_answer'] = q.pop('answer', 'n/a')
            q['model_answer'] = extract_single_letter(raw_text)
            q['kg_context_used'] = question_kg_map.get(f"Q{idx+1}")
            if idx == 2: q['audio_masked'] = True
            final_results.append(q)

        with open(save_path, 'w') as f:
            json.dump({"video_id": video_id_lower, "results": final_results}, f, indent=4)
            
        print(f"  [SUCCESS] Saved: {trans_id}")
        return True

async def run_evaluation():
    start_total = time.perf_counter()
    qa_files = sorted([f for f in os.listdir(JSON_ROOT) if f.endswith('_qa.json')])
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    async with aiohttp.ClientSession() as session:
        if not await check_server_health(session):
            print("!!! vLLM Server offline."); return

        for idx, qa_filename in enumerate(qa_files, 1):
            v_id = qa_filename.replace('_qa.json', '').strip()
            if v_id.lower() not in VIDEO_MAP: continue
            
            v_subdir = OUTPUT_DIR / v_id
            v_subdir.mkdir(parents=True, exist_ok=True)
            
            with open(JSON_ROOT / qa_filename, 'r') as f:
                data_list = json.load(f)

            print(f"[{idx}/{len(qa_files)}] Folder: {v_id}")
            tasks = [process_transition(session, semaphore, entry, v_id.lower(), v_subdir) for entry in data_list]
            await asyncio.gather(*tasks)

    print(f"\nFinal Runtime: {(time.perf_counter() - start_total)/60:.2f}m")

if __name__ == "__main__":
    asyncio.run(run_evaluation())