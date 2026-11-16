"""Cross-episode memory for A4.

A ChromaDB collection of one-sentence lessons, embedded with all-MiniLM-L6-v2.
Lessons are keyed to nothing but their text: retrieval is by similarity to the
current instruction, so a lesson only helps if it generalises past the scene it
came from.  That is the whole hypothesis A4 tests.
"""
import os
from typing import List, Optional


class LessonMemory:
    def __init__(self, path: str, collection: str = "lessons",
                 embed_model: str = "all-MiniLM-L6-v2") -> None:
        import chromadb
        from chromadb.utils import embedding_functions
        os.makedirs(path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
        self.col = self.client.get_or_create_collection(name=collection, embedding_function=self.ef)
        self._n = self.col.count()

    def add(self, lesson: str, source_instruction: str = "") -> None:
        self._n += 1
        self.col.add(ids=[f"lesson-{self._n}"], documents=[lesson],
                     metadatas=[{"from": source_instruction[:200]}])

    def search(self, query: str, k: int = 3) -> List[str]:
        if self.col.count() == 0:
            return []
        r = self.col.query(query_texts=[query], n_results=min(k, self.col.count()))
        docs = (r.get("documents") or [[]])[0]
        return [d for d in docs if isinstance(d, str)]

    def count(self) -> int:
        return self.col.count()

    def reset(self) -> None:
        name = self.col.name
        self.client.delete_collection(name)
        self.col = self.client.get_or_create_collection(name=name, embedding_function=self.ef)
        self._n = 0
