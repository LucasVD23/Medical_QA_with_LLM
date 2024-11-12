from gpt import GPT
from ragas_metrics import Faithulness

def rephraser_agent(state):
    llm_caller = GPT()

    query_prompt = """Rephrase the question using medical terminology to abstract the patient's specific symptoms and conditions
Question: {}"""

    original_question = state.get('original_question', '').strip()
    rephrased_question = state.get('rephrased_question', '').strip()
    options = state.get('options', '').strip()
    rag_information = state.get('rag_information', '').strip()
    cot_output = state.get('cot_output', '')
    faith_score = state.get('faith_score', '')

    rephrased_question = llm_caller.get_answer(query_prompt.format(original_question))


    return {
            'original_question' : original_question,
            'rephrased_question' : rephrased_question,
            'options': options,
            'rag_information' : rag_information,
            'cot_output' : cot_output,
            'faith_score' : faith_score
    }

def cot_agent(state):
    llm_caller = GPT()

    query_prompt = """Your role is to be a medical assistant for performing Q&A
in the medical domain. Given the following question and options, return the
correct option thinking step-by-step on how to get to the final answer.
Return the answer in the following format:
Answer: (option from A to E)"""


    original_question = state.get('original_question', '').strip()
    rephrased_question = state.get('rephrased_question', '').strip()
    options = state.get('options', '').strip()
    rag_information = state.get('rag_information', '').strip()
    cot_output = state.get('cot_output', '')
    faith_score = state.get('faith_score', '')


    question = rephrased_question if rephrased_question != '' else original_question

    if rag_information != ''.strip():
      query_prompt += """\nUse the following context to answer the question: {}
Question: {}
Options:{}
      """.format(rag_information, question, options)
    else:
      query_prompt += """
Question: {}
Options:{}
      """.format(question, options)

    cot_output = llm_caller.get_answer(query_prompt)

    return {
            'original_question' : original_question,
            'rephrased_question' : rephrased_question,
            'options': options,
            'rag_information' : rag_information,
            'cot_output' : cot_output,
            'faith_score' : faith_score
    }


def rag_agent(state, rag_module):
    llm_caller = GPT()

    relevant_prompt = """Given the following question and documents, extract only the documents that are relevant to the question
Question: {}
Documents: {}
    """

    useful_prompt = """Given the follow extract only the segments that are useful to the question
Question: {}
Documents: {}
"""


    original_question = state.get('original_question', '').strip()
    rephrased_question = state.get('rephrased_question', '').strip()
    options = state.get('options', '').strip()
    rag_information = state.get('rag_information', '').strip()
    cot_output = state.get('cot_output', '')
    answer_idx = state.get('answer_idx', '')
    faith_score = state.get('faith_score', '')

    rag_docs = rag_module.retrieve_documents(original_question, num_docs = 5)

    rag_information = "\n".join(str(item) for item in rag_docs)

    relevant = llm_caller.get_answer(relevant_prompt.format(original_question,
                                                                   rag_information))

    context = llm_caller.get_answer(useful_prompt.format(original_question,
                                                    relevant))


    return {
            'original_question' : original_question,
            'rephrased_question' : rephrased_question,
            'options': options,
            'rag_information' : context,
            'cot_output' : cot_output,
            'answer_idx' : answer_idx,
            'faith_score' : faith_score

    }

def faithfulness_agent(state):
    faithfulness_llm = Faithulness(GPT())

    original_question = state.get('original_question', '').strip()
    rephrased_question = state.get('rephrased_question', '').strip()
    options = state.get('options', '').strip()
    rag_information = state.get('rag_information', '').strip()
    answer = state.get('answer', '')
    faith_score = state.get('faith_score', '')

    question_plus_options = f'{original_question}\n{options}'
    # print(question_plus_options)
    faith_score = faithfulness_llm.evaluate(question_plus_options, answer, rag_information)

    return {
            'original_question' : original_question,
            'rephrased_question' : rephrased_question,
            'options': options,
            'rag_information' : rag_information,
            'answer' : answer,
            'faith_score' : faith_score

    }

