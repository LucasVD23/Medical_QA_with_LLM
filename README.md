# Deep Learning for NLP - Final Project
## Improving GPT-4o mini on the MedQA Dataset using Multi-Agent Framework

This repository contains the final project for the Deep Learning for NLP course at the State University of Campinas (Unicamp). The project investigates the use of a multi-agent approach to enhance the performance of the GPT-4o mini language model on the MedQA dataset, focusing on clinical question answering.

## Project Overview
In this project, we explore the use of specialized agents to support GPT-4o mini in better understanding and answering clinical questions. The goal is to address limitations in medical knowledge and context processing by incorporating domain-specific enhancements.

## Key Agents
The multi-agent approach employs the following specialized agents:

Query Rewriter: Reformulates the input query using medical terminology to improve comprehension.
Chain-of-Thought (CoT) Reasoner: Decomposes complex questions into logical steps to enable step-by-step reasoning.
FAISS Vectorstore Retriever: Searches for relevant medical passages from textbooks and selects the most pertinent segments.
Faithfulness Validator: Assesses the generated answer's faithfulness to the retrieved context, ensuring it aligns with the reference material.

## Methodology
Our approach draws inspiration from:

Wang, Y., Xueguang, M., & Wenhu, C. (2023). *Augmenting black-box LLMs with medical textbooks for clinical question answering*. arXiv preprint [arXiv:2309.02233](https://arxiv.org/abs/2309.02233).

The agents work in tandem to rewrite, analyze, retrieve, and validate information, which ultimately strengthens the language model's accuracy and faithfulness in clinical domains.


# Project Setup

This guide provides instructions on setting up the project environment, including downloading necessary datasets, installing dependencies, and running tests.

## 1. Download the Dataset

Download the dataset from the following link:
[MedQA Dataset](https://github.com/jind11/MedQA).

Once downloaded, place the dataset in the root directory of this repository.

## 2. Install Requirements

Install the required packages listed in `requirements.txt` by running:

```bash
pip install -r requirements.txt
```
unzip medical_textbooks.zip

## 3. Unzip the vectorstore

```bash
unzip medical_textbooks.zip
````

## 4. Set OpenAI Key

```bash
export OPENAI_API_KEY="your_api_key_here"
```
## 5. Run Tests
After completing the setup, you can run the tests using the run_tests.sh script:

```bash
run_tests.sh
````

