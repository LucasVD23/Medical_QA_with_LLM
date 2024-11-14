import json
from langgraph.graph import StateGraph, END
from RAG import RAG
from gpt import GPT
from helpers import get_textbooks, test_model_without_agents, test_workflow
from ragas_metrics import Faithulness
from multiagents.agents import cot_agent, rephraser_agent, rag_agent, faithfulness_agent
from multiagents.graph_state import GraphState
import random
import pandas as pd


path_questions ='data_clean/questions/US/test.jsonl'
with open(path_questions, 'r') as file:
    test_set = [json.loads(line) for line in file]

random.seed(15)
random.shuffle(test_set)

def run_test_routine():
    llm = GPT()

    #Without agents
    results_without_agents = test_model_without_agents(test_set, llm)
    results_without_agents_df = pd.DataFrame.from_dict(results_without_agents)
    results_without_agents_df.to_csv('results_without_agents.csv', index = None)

    #With CoT
    cot_workflow = StateGraph(GraphState)

    cot_workflow.add_node("cot_agent", cot_agent)
    cot_workflow.set_entry_point("cot_agent")
    cot_workflow.add_edge("cot_agent",END)

    results_with_cot = test_workflow(cot_workflow, test_set, test_type='cot_all')
    results_with_cot_df = pd.DataFrame.from_dict(results_with_cot)
    results_with_cot_df.to_csv('results_with_cot_all.csv', index = None)

    #With rewrite + CoT 

    re_write_workflow = StateGraph(GraphState)

    re_write_workflow.add_node("rephraser_agent", rephraser_agent)
    re_write_workflow.add_node("cot_agent", cot_agent)
    re_write_workflow.set_entry_point("rephraser_agent")
    re_write_workflow.add_edge("rephraser_agent","cot_agent")
    re_write_workflow.add_edge("cot_agent",END)

    results_with_rephrasal = test_workflow(re_write_workflow, test_set, test_type='reph_all')

    results_with_rephrasal = pd.DataFrame.from_dict(results_with_rephrasal)
    results_with_rephrasal.to_csv('results_with_rephrasal_all.csv', index = None)

    #With RAG + Cot

    rag_workflow = StateGraph(GraphState)

    rag_workflow.add_node("cot_agent", cot_agent)
    rag_workflow.add_node("rag_agent", rag_agent)
    rag_workflow.add_node("faithfulness_agent", faithfulness_agent)

    rag_workflow.set_entry_point("rag_agent")
    rag_workflow.add_edge("rag_agent","cot_agent")
    rag_workflow.add_edge("cot_agent","faithfulness_agent")
    rag_workflow.add_edge("faithfulness_agent",END)
 
    results_with_rag = test_workflow(rag_workflow, test_set, test_type = 'rag_all')

    results_with_rag_df = pd.DataFrame.from_dict(results_with_rag)
    results_with_rag_df.to_csv('results_with_rag_all.csv', index = None)

    #With Re-write + RAG + CoT

    reph_rag_workflow = StateGraph(GraphState)
    reph_rag_workflow.add_node("rephraser_agent", rephraser_agent)
    reph_rag_workflow.add_node("cot_agent", cot_agent)
    reph_rag_workflow.add_node("rag_agent", rag_agent)
    reph_rag_workflow.add_node("faithfulness_agent", faithfulness_agent)

    reph_rag_workflow.set_entry_point("rephraser_agent")
    reph_rag_workflow.add_edge("rephraser_agent","rag_agent")
    reph_rag_workflow.add_edge("rag_agent","cot_agent")
    reph_rag_workflow.add_edge("cot_agent","faithfulness_agent")
    reph_rag_workflow.add_edge("faithfulness_agent",END)
    results_reph_rag = test_workflow(reph_rag_workflow, test_set,test_type =  'reph_rag_all')

    results_reph_rag_df = pd.DataFrame.from_dict(results_reph_rag)
    results_reph_rag_df.to_csv('results_reph_rag_all.csv', index = None)

def main():
    print("Initializing test routine...")
    run_test_routine()
    print("Test routine completed. Results saved.")

if __name__ == "__main__":
    main()