from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import time
import json
import concurrent.futures
import os
from openai import OpenAI
import threading

# Configuration
api_key = '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'
base_url = 'YOUR BASE URL'
num_workers = 10  # Adjust based on system capacity

client = OpenAI(api_key=api_key, base_url=base_url)
df = pd.read_csv('news_dataset.csv', delimiter=';')

# Thread lock to ensure safe file writing
lock = threading.Lock()

# Track progress
progress_file = "progress.txt"
updated_file = "updated.csv"

# Load previously completed rows if they exist
if os.path.exists(updated_file):
    completed_df = pd.read_csv(updated_file, delimiter=';')
    completed_indices = set(completed_df.index)  # Indices of processed rows
else:
    completed_indices = set()

def get_response(prompt, max_tokens=40, max_retries=3):
    """Retries API call up to max_retries times if it fails."""
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model="YOUR QWEN MODEL",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=1e-6
            )
            if response and hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            else:
                print(f"API returned an empty or invalid response. Attempt {attempt+1}/{max_retries}")
                time.sleep(2)  # Retry delay
        except Exception as e:
            print(f"Error on attempt {attempt+1}/{max_retries}: {e}")
            time.sleep(2)  # Wait before retrying
        attempt += 1
    return None  # Return None if all retries fail

def get_prompt(news, political_perspective):
    """Generates the prompt based on the political perspective."""
    prompt_template = """You should decide if a news article is “True” or “False” from the view point of a {party} supporter. What would a {party} supporter think of the following news article?

    The news: {news}

    You must determine whether it is True (1) or False (0). Respond in the following format:

    <0 or 1>. Reasoning Explanation.

    Your explanation must be concise, up to 40 tokens. Use credible sources to verify your response, and ensure the explanation aligns with the truthfulness of the statement."""
    
    if political_perspective == 'Neutral':
        prompt_template = f"""I will provide a news statement below.

        The news: {news}

        You must determine whether it is True (1) or False (0). Respond in the following format:

        <0 or 1>. Reasoning Explanation.

        Your explanation must be concise, up to 30 tokens. Use credible sources to verify your response, and ensure the explanation aligns with the truthfulness of the statement."""
    
    parties = {
        "Democrat": "democrat",
        "Republican": "republican",
        "Neutral": "neutral"
    }
    
    return prompt_template.format(party=parties[political_perspective], news=news)

def extract_response(response):
    """Extracts the validation (0 or 1) and explanation from the model output."""
    if response is None:
        return None, "Error: No response from API"

    pattern = r"<?(\d)>?\.\s*(.*)"
    match = re.search(pattern, response)
    if match:
        return int(match.group(1)), match.group(2).strip()
    
    return None, response  # Return raw response if format doesn't match

def process_row(args):
    """Processes a single row in parallel."""
    idx, row, iter_count = args

    # Skip already completed rows
    if idx in completed_indices:
        print(f"Skipping row {idx}, already processed.")
        return None

    news = row['News']
    results = []
    
    for perspective in ['Democrat', 'Republican', 'Neutral']:
        for i in range(iter_count):
            prompt = get_prompt(news, perspective)
            response = get_response(prompt)

            validation, explanation = extract_response(response)
            
            result = {
                'Index': idx,
                'News': news,
                'Perspective': perspective,
                'Iteration': i,
                'Validations': validation,
                'Explanations': explanation
            }
            results.append(result)

            # Save incrementally to avoid data loss
            with lock:
                pd.DataFrame([result]).to_csv('explanations.csv', mode='a', header=False, index=False)
    
    # Write progress to file
    with lock:
        with open(progress_file, "a") as f:
            f.write(f"{idx}\n")  # Store the processed index

    return idx, results  # Return index and results for updating counts

def run(iter_count, num_workers):
    """Runs the processing with parallel execution."""
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = [(idx, row, iter_count) for idx, row in df.iterrows() if idx not in completed_indices]
        for idx, results in tqdm(executor.map(process_row, tasks), total=len(tasks), desc="Processing rows in parallel"):
            if results is None:
                continue  # Skip rows that were already processed

            all_results.extend(results)

            # Update counts in the main dataframe
            true_counts = {persp: sum(1 for r in results if r['Validations'] == 1 and r['Perspective'] == persp) for persp in ['Democrat', 'Republican', 'Neutral']}
            false_counts = {persp: sum(1 for r in results if r['Validations'] == 0 and r['Perspective'] == persp) for persp in ['Democrat', 'Republican', 'Neutral']}

            df.at[idx, 'Count True Democrat'] = true_counts['Democrat']
            df.at[idx, 'Count False Democrat'] = false_counts['Democrat']
            df.at[idx, 'Count True Republican'] = true_counts['Republican']
            df.at[idx, 'Count False Republican'] = false_counts['Republican']
            df.at[idx, 'Count True Neutral'] = true_counts['Neutral']
            df.at[idx, 'Count False Neutral'] = false_counts['Neutral']

            # Save incrementally
            with lock:
                df.iloc[[idx]].to_csv('updated.csv', mode='a', header=False, index=False)

    # Final saving after all processes complete
    df.to_csv('updated.csv', index=False)
    print("Processing complete. Data saved.")

# Run with multiprocessing and resume support
iter_count = 10
run(iter_count=iter_count, num_workers=num_workers)