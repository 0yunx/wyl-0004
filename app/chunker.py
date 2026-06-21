import re
from typing import List
from .config import settings


class TextChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    @staticmethod
    def _split_by_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    @staticmethod
    def _split_by_paragraphs(text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def chunk(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text else []

        paragraphs = self._split_by_paragraphs(text)
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            sentences = self._split_by_sentences(paragraph)

            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    if len(sentence) > self.chunk_size:
                        words = sentence.split()
                        temp_chunk = ""
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 <= self.chunk_size:
                                temp_chunk += " " + word if temp_chunk else word
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = word
                        current_chunk = temp_chunk
                    else:
                        if self.chunk_overlap > 0 and chunks:
                            last_chunk = chunks[-1]
                            overlap_start = max(0, len(last_chunk) - self.chunk_overlap)
                            overlap_text = last_chunk[overlap_start:]
                            current_chunk = overlap_text + " " + sentence
                        else:
                            current_chunk = sentence

            current_chunk += "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        chunks = [c for c in chunks if c.strip()]
        return chunks
