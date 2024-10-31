import torch
from abc import ABC, abstractmethod
import re
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

class Metric(ABC):

    def __init__(self, model):
        self.model = model

    #Obriga a implementação do evaluate
    @abstractmethod
    def evaluate(self, question, context, answer):
        raise NotImplementedError

class Faithulness(Metric):
    def __init__(self, model):  # Add model as an argument
        super().__init__(model)  # Pass the model to the parent class

    def __process_statements(self, statements):
        """
        Função para extrair afirmações da LLM
        """
        #Expressão regular para capturar statemnts
        pattern = r"Statement \d+:\s*(.+)"

        matches = re.findall(pattern, statements)

        #Caso a resposta não esteja de acordo com o formato especificado
        if not matches:
          print(f"No statements found in the response: {statements}")

        return matches

    def __construct_verdict_prompt(self, statement_list, context):
        prompt = (
            f"Consider the given context and following\n"
            f"statements, then determine whether they\n"
            f"are supported by the information present\n"
            f"in the context. Provide a brief explanation for each statement before arriving\n"
            f"at the verdict (Yes/No). Provide a final\n"
            f"verdict for each statement in order at the\n"
            f"end in the given format. Do not deviate\n"
            f"from the specified format.\n\n"
            f"Context: {context}\n\n"
        )

        #Adiciona as afirmações no prompt
        for i, statement in enumerate(statement_list, 1):
            prompt += f"Statement {i}: {statement}\n"
        #Adiciona o formato de veredito
        prompt += "\nFinal verdict format:\n"
        for i in range(1, len(statement_list) + 1):
            prompt += f"Verdict {i}: Yes/No\n"

        return prompt

    def __process_verdicts(self, verdicts):
        """
        Função para processar as respostas com vereditos
        """
        #Expressão regular para capturar vereditos
        verdict_pattern = re.findall(r"Verdict \d+:\s*(Yes|No)", verdicts)

        #Converte vereditos em 1 e 0
        binary_verdicts = [1 if v == 'Yes' else 0 for v in verdict_pattern]

        if not binary_verdicts:
            print(f"No verdicts found in the response: {verdicts}")
            return 0

        return sum(binary_verdicts) / len(binary_verdicts)

    def evaluate(self, question, answer, context):
        #Primeiro prompt para pegar as afirmações
        get_statements_prompt = f"""
        Given a question and answer, create one
        or more statements from each sentence
        in the given answer.

        Return the response in the following format:

        Statement 1: statement_1
        ...
        Statement n: statement_n

        Do not deviate from the specified format.

        question: {question}
        answer: {answer}
        """
        #Uso do modelo para capturar os statements
        statements = self.model.get_answer(get_statements_prompt)
        #Processametno das saidas
        processed_statements = self.__process_statements(statements)

        #Construir o segundo prompt
        faithfulness_prompt = self.__construct_verdict_prompt(processed_statements, context)
        #Vereditos da LLM
        verdicts = self.model.get_answer(faithfulness_prompt)

        average_score = self.__process_verdicts(verdicts)

        return average_score


class AnswerRelevance(Metric):
    def __init__(self, model):
        super().__init__(model)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)

    def evaluate(self, question, answer, n=5):

        # Gera N questões a partir da resposta
        generated_questions = self.__generate_questions_from_answer(answer, n)

        #Calcula os emebeddings
        original_question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
        generated_question_embeddings = self.embedding_model.encode(generated_questions, convert_to_tensor=True)

        #Calcula similaridade por cosseno
        similarities = self.__calculate_similarities__(original_question_embedding, generated_question_embeddings)

        #Média das similaridades
        average_relevance_score = torch.mean(torch.tensor(similarities)).item()

        return average_relevance_score

    def __generate_questions_from_answer(self, answer, n):
        generated_questions = []
        for _ in range(n):
            #Prompt de geração das perguntas
            generate_question_prompt = f"Generate a question for the given answer.\nanswer: {answer}"
            question = self.model.get_answer(generate_question_prompt)
            generated_questions.append(question.strip())

        return generated_questions

    def __calculate_similarities__(self, original_embedding, generated_embeddings):

        original_embedding = original_embedding.unsqueeze(0)

        similarities = F.cosine_similarity(generated_embeddings, original_embedding, dim=1).tolist()

        return similarities

class ContextRelevance(Metric):
    def __init__(self, model):
        super().__init__(model)

    def evaluate(self, question, context):
        # Calculate total sentences across all documents in the context
        total_sentences = sum(len(self.__split_into_sentences(doc)) for doc in context)

        # Extract relevant sentences from the context
        extracted_sentences = self.__extract_relevant_sentences(question, context)

        # Calculate relevance score as the ratio of relevant sentences to total sentences
        if "Insufficient Information" in extracted_sentences:
            relevance_score = 0.0
        else:
            relevance_score = len(extracted_sentences) / total_sentences

        return relevance_score, extracted_sentences

    def __extract_relevant_sentences(self, question, context):
        # Join all documents into a single text for extraction
        context_text = " ".join(context)

        # Prompt for sentence extraction
        extract_prompt = f"""
        Please extract relevant sentences from the provided context that can potentially help answer the following question.
        If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information".
        While extracting candidate sentences, you’re not allowed to make any changes to sentences from the given context.

        question: {question}
        context: {context_text}
        """

        extracted_sentences = self.model.get_answer(extract_prompt)

        if "Insufficient Information" in extracted_sentences:
            return ["Insufficient Information"]
        else:
            # Process extracted sentences into a list
            return self.__process_extracted_sentences(extracted_sentences)

    def __process_extracted_sentences(self, extracted_sentences):
        # Split extracted text into sentences, stripping whitespace
        list_of_sentences = [sentence.strip() for sentence in extracted_sentences.split('. ') if sentence]

        return list_of_sentences

    def __split_into_sentences(self, document):
        return [sentence.strip() for sentence in document.split('. ') if sentence]