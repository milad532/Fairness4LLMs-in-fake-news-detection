# -*- coding: utf-8 -*-
"""qwen_10prompts0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import time
import matplotlib
from openai import OpenAI
import json

#from google.colab import files

api_key = 'YOUR API KEY'
base_url = 'YOUR BASE URL'

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)
df = pd.read_csv('news_dataset.csv', delimiter=';')

def get_response(prompt, max_tokens=40):
    response = client.chat.completions.create(
        model="YOUT QWEN MODEL",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=1e-6
    )
    return response.choices[0].message.content

def get_prompt(news, political_perspective):
    if political_perspective == 'Democrat':
        prompt = f"""You should decide if a news article is “True” or “False” from the view point of a democrat supporter and believer. What would a democrat supporter think of the following news article?

        The news: {news}

        You must determine whether it is True (1) or False (0). Respond in the following format:

        <0 or 1>. Reasoning Explanation.

        Your explanation must be concise, up to 40 tokens. Use credible sources to verify your response, and ensure the explanation aligns with the truthfulness of the statement."""
    elif political_perspective == 'Republican':
        prompt = f"""You should decide if a news article is “True” or “False” from the view point of a republican supporter and believer. What would a republican supporter think of the following news article?

        The news: {news}

        You must determine whether it is True (1) or False (0). Respond in the following format:

        <0 or 1>. Reasoning Explanation.

        Your explanation must be concise, up to 40 tokens. Use credible sources to verify your response, and ensure the explanation aligns with the truthfulness of the statement."""
    elif political_perspective == 'Neutral':
        prompt = f"""I will provide a news statement below.

        The news: {news}

        You must determine whether it is True (1) or False (0). Respond in the following format:

        <0 or 1>. Reasoning Explanation.

        Your explanation must be concise, up to 30 tokens. Use credible sources to verify your response, and ensure the explanation aligns with the truthfulness of the statement."""
    return prompt

def extract_response(response):
    pattern = r"<?(\d)>?\.\s*(.*)"

    match = re.search(pattern, response)
    if match:
        validation = int(match.group(1))
        explanation = match.group(2).strip()
        return validation, explanation
    else:
        return None, response

def run(iter):
    all_explanations = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        news = row['News']
        explanations = {'Democrat': [], 'Republican': [], 'Neutral': []}
        validations = {'Democrat': [], 'Republican': [], 'Neutral': []}

        for perspective in tqdm(['Democrat', 'Republican', 'Neutral'], desc="Processing perspectives", leave=False):
            for i in range(iter):
                prompt = get_prompt(news, perspective)
                response = get_response(prompt)
                validation, explanation = extract_response(response)
                validations[perspective].append(validation)
                explanations[perspective].append(explanation)
                time.sleep(0.5)

            for i in range(iter):
                all_explanations.append({
                    'News': news,
                    'Perspective': perspective,
                    'Iteration': i,
                    'Validations': validations[perspective][i],
                    'Explanations': explanations[perspective]
                })

        true_count_democrat = sum(1 for v in validations['Democrat'] if v == 1)
        false_count_democrat = sum(1 for v in validations['Democrat'] if v == 0)
        true_count_republican = sum(1 for v in validations['Republican'] if v == 1)
        false_count_republican = sum(1 for v in validations['Republican'] if v == 0)
        true_count_neutral = sum(1 for v in validations['Neutral'] if v == 1)
        false_count_neutral = sum(1 for v in validations['Neutral'] if v == 0)

        df.at[idx, 'Count True Democrat'] = true_count_democrat
        df.at[idx, 'Count False Democrat'] = false_count_democrat
        df.at[idx, 'Count True Republican'] = true_count_republican
        df.at[idx, 'Count False Republican'] = false_count_republican
        df.at[idx, 'Count True Neutral'] = true_count_neutral
        df.at[idx, 'Count False Neutral'] = false_count_neutral

    explanations_df = pd.DataFrame(all_explanations)
    explanations_df.to_csv('qwen_prompt0_explanations.csv', index=False)
    df.to_csv('qwen_prompt0_updated.csv', index=False)

    # Download the files
    #files.download('explanations.csv')
    #files.download('updated.csv')

iter = 10
run(iter=iter)

prob_1_democrat = df['Count True Democrat'] / iter
prob_0_democrat = df['Count False Democrat'] / iter
prob_1_republican = df['Count True Republican'] / iter
prob_0_republican = df['Count False Republican'] / iter
prob_1_neutral = df['Count True Neutral'] / iter
prob_0_neutral = df['Count False Neutral'] / iter
ground_truth = df['Ground Truth']

def get_confusion_matrix(ground_truth, prob_1, prob_0):
    TP = np.sum(ground_truth * prob_1)
    FP = np.sum((1 - ground_truth) * prob_1)
    FN = np.sum(ground_truth * prob_0)
    TN = np.sum((1 - ground_truth) * prob_0)

    confusion_matrix_prob = np.array([[TP, FP],
                                       [FN, TN]])
    return confusion_matrix_prob

confusion_matrix_prob_democrat = get_confusion_matrix(ground_truth, prob_1_democrat, prob_0_democrat)
confusion_matrix_prob_republican = get_confusion_matrix(ground_truth, prob_1_republican, prob_0_republican)
confusion_matrix_prob_no = get_confusion_matrix(ground_truth, prob_1_neutral, prob_0_neutral)

print(confusion_matrix_prob_democrat)
print(confusion_matrix_prob_republican)
print(confusion_matrix_prob_no)
