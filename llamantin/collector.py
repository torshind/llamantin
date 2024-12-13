import asyncio
import os

from faiss import IndexFlatL2
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from unstructured.partition.auto import partition
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class Collector:
    def __init__(self, directory: str, db_path: str = "vector_db.faiss"):
        self.directory = directory
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if os.path.exists(db_path):
            self.vector_db = FAISS.load_local(
                folder_path=db_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.is_db_loaded = True
        else:
            self.vector_db = FAISS(
                embedding_function=self.embeddings,
                index=IndexFlatL2(len(self.embeddings.embed_query("embedding size"))),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self.is_db_loaded = False
        self.observer = Observer()
        self.event_handler = FileSystemEventHandler()
        self.event_handler.on_modified = self.on_modified
        self.event_handler.on_created = self.on_created

    async def initialize_database(self):
        if not self.is_db_loaded:
            await self.crawl_directory(self.directory)
            self.vector_db.save_local(folder_path=self.db_path)
            self.is_db_loaded = True
            print("Database initialized")

    async def crawl_directory(self, directory: str):
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                await self.process_file(file_path)

    async def process_file(self, file_path: str):
        try:
            elements = partition(file_path)
            content = "\n".join([element.text for element in elements if element.text])
            if not content:
                raise ValueError("No content extracted")
        except Exception as e:
            print(f"Unsupported format - {file_path}: {e}")
            return

        document = Document(page_content=content, metadata={"path": file_path})
        await self.vector_db.aadd_documents(documents=[document])
        self.vector_db.save_local(folder_path=self.db_path)

    def on_modified(self, event):
        if not event.is_directory:
            asyncio.run(self.process_file(event.src_path))

    def on_created(self, event):
        if not event.is_directory:
            asyncio.run(self.process_file(event.src_path))

    def start(self):
        self.observer.schedule(self.event_handler, self.directory, recursive=True)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()
