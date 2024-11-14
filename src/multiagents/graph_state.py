
from typing import TypedDict, Optional

class GraphState(TypedDict):
    original_question: Optional[str] =  None,
    options: Optional[str] = None,
    rephrased_question: Optional[str] =  None,
    rag_information: Optional[str] = None,
    cot_output: Optional[str] = None,
    faith_score: Optional[float] = None