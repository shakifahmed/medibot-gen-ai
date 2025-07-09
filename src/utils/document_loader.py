from config.config import Config
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self):
        self.config = Config()

    def load_document(self):
        dir_loader = DirectoryLoader(
            path=self.config.DATA_PATH,
            glob=self.config.PDF_GLOB,
            loader_cls=PyPDFLoader
        )
        docs = dir_loader.load()
        return docs
    
    def create_chunks(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        docs = self.load_document()
        chunks = splitter.split_documents(documents=docs)
        return chunks