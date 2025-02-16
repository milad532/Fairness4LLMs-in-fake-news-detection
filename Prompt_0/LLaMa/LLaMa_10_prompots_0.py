from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import time
import json
import concurrent.futures
from openai import OpenAI
import threading

# Configuration
api_key = 'YOUR API KEY'
base_url = 'YOUR BASE URL'
num_workers = 32  # Number of concurrent API calls

client = OpenAI(api_key=api_key, base_url=base_url)
df = pd.read_csv('news_dataset.csv', delimiter=';')

# Thread lock to ensure safe file writing
lock = threading.Lock()

def get_response(prompt, max_tokens=40, max_retries=3):
    """Retries API call up to max_retries times if it fails."""
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model="YOUR LLAMA MODEL",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=1e-6
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error on attempt {attempt+1}/{max_retries}: {e}")
            attempt += 1
            time.sleep(2)  # Wait before retrying
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
    pattern = r"<?(\d)>?\.\s*(.*)"
    match = re.search(pattern, response)
    if match:
        return int(match.group(1)), match.group(2).strip()
    return None, response  # Return raw response if format doesn't match

def process_row(args):
    """Processes a single row in parallel."""
    idx, row, iter_count = args
    news = row['News']
    results = []
    
    for perspective in ['Democrat', 'Republican', 'Neutral']:
        for i in range(iter_count):
            prompt = get_prompt(news, perspective)
            response = get_response(prompt)
            if response is not None:
                validation, explanation = extract_response(response)
            else:
                validation, explanation = None, "Error in response"

            result = {
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
    
    return idx, results  # Return index and results for updating counts

def run(iter_count, num_workers):
    """Runs the processing with parallel execution."""
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = [(idx, row, iter_count) for idx, row in df.iterrows()]
        for idx, results in tqdm(executor.map(process_row, tasks), total=len(df), desc="Processing rows in parallel"):
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

# Run with multiprocessing
iter_count = 10
run(iter_count=iter_count, num_workers=num_workers)