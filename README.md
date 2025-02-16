# Fake News Detection Using LLMs



## 📌 Description

This repository contains the implementation of the paper *"Toward Fair and Effective Fake News Detection: Assessing Large Language Models."* The project focuses on evaluating the fairness and efficiency of Large Language Models (LLMs) in detecting fake news using a dataset of news articles classified by political leaning.

## 🚀 Features

- Analysis of LLM biases in news classification
- Evaluation of model fairness and accuracy
- Benchmarking multiple LLMs (GPT-4o, LLaMa, Qwen, Deepseek)
- News leaning classification (Democrat, Republican, Neutral, Varies)
- Fake news detection using a labeled dataset



## 📖 Usage

use the `news_dataset.csv` and `news_leaning_dataset.csv` as dataset and `log_processor.py` as processor for the outputs of each LLM.

## 🛠️ Technologies Used

- Python
- open-ai
- Scikit-learn
- Pandas & NumPy

## 📊 Dataset

The dataset consists of news articles labeled with political leanings and fact-checking results. The files include:

- **news_dataset.csv**: Contains raw news articles with metadata with labeled POV.
- **news_leaning_dataset.csv**: Labels news articles as Democrat, Republican, Neutral, or Varies with Labeled leanings.

