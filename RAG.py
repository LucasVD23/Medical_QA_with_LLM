from helpers import get_textbooks
import torch
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class RAG():
    def __init__(self, model_name, vector_store_path) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_kwargs = {'device': device}
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs)
        self.vector_store_path = vector_store_path


    def create_vectorstore(self,knowledge_base_path, text_splitter = None):

        if text_splitter is None:
          text_splitter = SemanticChunker(self.embedding_model)
        textbooks = get_textbooks(knowledge_base_path)
        chunks = text_splitter.create_documents(textbooks)

        faiss_store  = FAISS.from_documents(chunks, embedding=self.embedding_model)
        faiss_store.save_local(self.vector_store_path)

    def retrieve_documents(self, query, num_docs = 5):
        faiss_store = FAISS.load_local(self.vector_store_path,
                                self.embedding_model,
                                allow_dangerous_deserialization = True)

        similar_docs = faiss_store.similarity_search(query, k=num_docs)
        similar_docs = [doc.page_content for doc in similar_docs]

        return similar_docs