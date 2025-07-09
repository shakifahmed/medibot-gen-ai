from src.embeddings.vector_store import VectorStore
from src.llm.chat_model import ChatModel
from src.retrieval.retriever import Retriever


def main():
    database = VectorStore()
    llm = ChatModel()
    

    vector_store = database.get_vector_store()
    retriever = Retriever(vector_store=vector_store)
    query = "Give Summary of 'Abstract'."
    context = retriever.retriever_context(query=query)
    answer, _ = llm.generate_answer(query=query, context=context)
    print("answer:\n", answer.content)

if __name__=="__main__":
    main()
