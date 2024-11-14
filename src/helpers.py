import os
import random
import json
from tqdm import tqdm
import pandas as pd

def get_textbooks(dataset_path):
    textbooks = []
    for book in os.listdir(dataset_path + 'textbooks/en/'):
        file = open(dataset_path + 'textbooks/en/' + book, 'r')
        textbooks.append('\n'.join(file.readlines()))

    return textbooks

def get_dataset(dataset_path):
    with open(dataset_path, 'r') as file:
        dataset = [json.loads(line) for line in file]

    random.seed(15)
    random.shuffle(dataset)
    return dataset

def test_model_without_agents(test_set,llm):
    query_prompt = """Given the following question and options, return the
    Return the answer in the following format:
    Answer: (option from A to E)

    Question: {}
    Options:{}"""

    test_results = {'questions':[],
                    'options':[],
                    'answer_idx': [],
                    'llm_answer': []}

    for item in test_set:
        question = item['question']
        options = item['options']
        answer_idx = item['answer_idx']



        llm_answer = llm.get_answer(query_prompt.format(question, options))
        print(query_prompt.format(question, options))
        print(llm_answer)
        test_results['questions'].append(question)
        test_results['options'].append(options)
        test_results['answer_idx'].append(answer_idx)
        test_results['llm_answer'].append(llm_answer.split('Answer:')[1].strip())

    return test_results

def load_checkpoint(filepath):
    """Loads checkpoint data from CSV and returns it as a dictionary."""
    df = pd.read_csv(filepath)
    return {
        'questions': df['questions'].tolist(),
        'options': df['options'].tolist(),
        'answer_idx': df['answer_idx'].tolist(),
        'llm_answer': df['llm_answer'].tolist(),
        'cot_output': df['cot_output'].tolist(),
        'faithfulness': df['faithfulness'].tolist(),
        'rag_information': df['rag_information'].tolist()
    }, len(df)


def test_workflow(workflow, test_set, test_type='all', checkpoint_path = None):
    # Load last checkpoint if available
    if checkpoint_path is not None:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        test_results, start_index = load_checkpoint(checkpoint_path)
  
    else:
        print("Starting a new experiment.")
        test_results = {
            'questions': [],
            'options': [],
            'answer_idx': [],
            'llm_answer': [],
            'cot_output': [],
            'faithfulness': [],
            'rag_information': []
        }
        start_index = 0

    app = workflow.compile()
    checkpoint_interval = max(1, len(test_set) // 10)  # Calculate 10% interval

    # Continue from the last processed sample
    for i, item in enumerate(tqdm(test_set[start_index:], desc="Testing workflow"), start=start_index + 1):
        question = item['question']
        options = item['options']
        answer_idx = item['answer_idx']
        state = {
            'original_question': question,
            'rephrased_question': '',
            'options': str(options),
            'rag_information': '',
            'cot_output': '',
            'faith_score': 0
        }

        conversation = app.invoke(state)
        llm_answer = conversation['cot_output'].split('Answer:')[-1].strip()

        test_results['questions'].append(question)
        test_results['options'].append(options)
        test_results['answer_idx'].append(answer_idx)
        test_results['llm_answer'].append(llm_answer)
        test_results['cot_output'].append(conversation['cot_output'])
        test_results['rag_information'].append(conversation.get('rag_information', ''))
        test_results['faithfulness'].append(conversation['faith_score'])

        # Save checkpoint every 10% of the process
        if i % checkpoint_interval == 0:
            df = pd.DataFrame(test_results)
            df.to_csv(f'test_results_checkpoint_{test_type}_{i // checkpoint_interval}.csv', index=False)

    # Save the final results
    df = pd.DataFrame(test_results)
    df.to_csv(f'test_results_final_{test_type}.csv', index=False)

    return test_results