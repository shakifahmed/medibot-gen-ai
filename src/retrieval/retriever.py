from langchain.retrievers import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from src.embeddings.vector_store import VectorStore
from config.config import Config


class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.config = Config()

    def setup_retriever(self):
        compressor = LLMChainExtractor.from_llm(
            llm=ChatGoogleGenerativeAI(model=self.config.LLM_MODEL)
        )

        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(
                search_kwargs={'k':self.config.RETRIEVAL_K}
            ),
            llm=ChatGoogleGenerativeAI(model=self.config.LLM_MODEL)
        )

        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=mq_retriever
        )

        return retriever

    def retriever_context(self, query):
        retriever = self.setup_retriever()
        retrieved_docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context