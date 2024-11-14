# Deep Learning for NLP - Final Project
## Investigating the improvement of GPT-4o mini on the MedQA Dataset using a Multi-Agent Framework

This repository contains the final project for the Deep Learning for NLP course at the State University of Campinas (Unicamp). The project investigates the use of a multi-agent approach to enhance the performance of the GPT-4o mini language model on the MedQA dataset, focusing on clinical question answering.

## Project Overview
In this project, we explore the use of specialized agents to support GPT-4o mini in better understanding and answering clinical questions. The goal is to address limitations in medical knowledge and context processing by incorporating domain-specific enhancements.

## Key Agents
The multi-agent approach employs the following specialized agents:

**Query Rewriter**: Reformulates the input query using medical terminology to improve comprehension.

**Chain-of-Thought (CoT) Reasoner**: Decomposes complex questions into logical steps to enable step-by-step reasoning.

**FAISS Vectorstore Retriever**: Searches for relevant medical passages from textbooks and selects the most pertinent segments.

**Faithfulness Validator**: Assesses the generated answer's faithfulness to the retrieved context, ensuring it aligns with the reference material.

## Methodology
Our approach draws inspiration by:

Wang, Y., Xueguang, M., & Wenhu, C. (2023). *Augmenting black-box LLMs with medical textbooks for clinical question answering*. arXiv preprint [arXiv:2309.02233](https://arxiv.org/abs/2309.02233).

The agents work in tandem to rewrite, analyze, retrieve, and validate information, which ultimately strengthens the language model's accuracy and faithfulness in clinical domains.

## Results
After implementing the agents and conducting preliminary tests, the final configurations were evaluated again using the full dataset of 1,271 samples. The obtained accuracy metrics for each strategy were as follows:

  - Direct Questioning (No Agents): 68.89%
  - Chain of Thought (CoT) Agent: 71.95%
  - Rewriting Agents + CoT: 67.16%
  - Retrieval-Augmented Generation (RAG) + CoT Agent: 68.65%
  - Rewriting Agent + RAG + CoT: 68.34%

These results demonstrate that the highest accuracy was achieved by the Chain of Thought (CoT) Agent strategy, with 71.95%. This suggests that using CoT reasoning without additional rewriting or retrieval steps was the most effective approach for this dataset. Further research could explore optimizing these configurations or combining methods in a way that leverages the strengths of the Rewrite, RAG and CoT approaches.


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

