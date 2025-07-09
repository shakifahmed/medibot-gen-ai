from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.utils.document_loader import DocumentLoader
from config.config import Config

class VectorStore:
    def __init__(self):
        self.config = Config()
        self.embedding = GoogleGenerativeAIEmbeddings(
            model=self.config.EMBEDDING_MODEL
        )

        self.dimension = len(self.embedding.embed_query("hi"))
        self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
        self.vector_store = None

    def create_index_if_not_exists(self):
        # assume index is not created
        index_created = False

        if not self.pc.has_index(self.config.INDEX_NAME):
            self.pc.create_index(
                name=self.config.INDEX_NAME,
                dimension=self.dimension,
                metric=self.config.METRIC,
                spec=ServerlessSpec(
                    cloud=self.config.PINECONE_CLOUD,
                    region=self.config.PINECONE_REGION
                )
            )

            index_created = True
            print("created new databse index")
        
        return index_created
    
    def document_insert_to_vector_store(self, chunks):
        try:
            self.vector_store = PineconeVectorStore.from_documents(
                documents=chunks,
                index_name=self.config.INDEX_NAME,
                embedding=self.embedding

            )
            print("data inseted using from_documents()")

        except:
            print("trying batch insert...")

            self.vector_store = PineconeVectorStore(
                index_name=self.config.INDEX_NAME,
                embedding=self.embedding
            )
            batch_size = self.config.BATCH_SIZE
            for i in range(0,len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self.vector_store.add_documents(documents=batch)
            print("data inserted using batch upload")

    def setup_vector_store(self):
        try:
            index_created = self.create_index_if_not_exists()
            index = self.pc.Index(self.config.INDEX_NAME)

            if index_created:
                doc_loader = DocumentLoader()
                chunks = doc_loader.create_chunks()

                #insert documents
                self.document_insert_to_vector_store(chunks)
                stats = index.describe_index_stats()
                print("chunks created and inserted to the vector database.")
                print("new database index stats:\n", stats)

            else:
                self.vector_store = PineconeVectorStore.from_existing_index(
                    index_name=self.config.INDEX_NAME,
                    embedding=self.embedding
                )
                stats = index.describe_index_stats()
                print("exiting index stats:\n",stats)

        except Exception as e:
            print(f"failed to connect pincone index: {e}")
            raise

    def get_vector_store(self):
        if self.vector_store is None:
            self.setup_vector_store()
        return self.vector_store
