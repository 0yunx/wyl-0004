"""Rule-based text chunker with paragraph → sentence → word fallback.

Chunking strategy::

    1. Split the document into paragraphs (double-newline boundaries).
    2. Within each paragraph, split into sentences (punctuation boundaries
       covering both CJK and Latin sentence terminators).
    3. Greedily accumulate sentences until ``chunk_size`` is exceeded.
    4. If a single sentence exceeds ``chunk_size``, split it by whitespace.
    5. When starting a new chunk, optionally prepend up to
       ``chunk_overlap`` characters from the previous chunk to preserve
       cross-chunk context.

This multi-level fallback ensures that no input text is lost, regardless
of how the source document is formatted.
"""

import re
from typing import List
from .config import settings


class TextChunker:
    """Split plain text into overlapping, size-bounded chunks.

    Args:
        chunk_size: Maximum characters per chunk (default from ``settings.CHUNK_SIZE``).
        chunk_overlap: Characters of overlap between consecutive chunks
            (default from ``settings.CHUNK_OVERLAP``).
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None) -> None:
        self.chunk_size: int = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap: int = chunk_overlap or settings.CHUNK_OVERLAP

    @staticmethod
    def _split_by_sentences(text: str) -> List[str]:
        """Split *text* at sentence-ending punctuation (CJK + Latin).

        Uses a lookbehind regex so the delimiter stays attached to the
        preceding sentence.
        """
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _split_by_paragraphs(text: str) -> List[str]:
        """Split *text* at blank-line boundaries."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk(self, text: str) -> List[str]:
        """Produce a list of text chunks from *text*.

        The algorithm:

        1. If the whole text fits in one chunk, return it as-is.
        2. Otherwise, iterate paragraphs → sentences, accumulating into
           the current chunk until ``chunk_size`` is exceeded.
        3. Overflow sentences that exceed ``chunk_size`` alone are split
           by whitespace (word-level).
        4. When a new chunk starts, up to ``chunk_overlap`` characters
           from the tail of the previous chunk are prepended.

        Args:
            text: Full document text.

        Returns:
            Non-empty list of chunk strings, each ≤ ``chunk_size`` characters.
        """
        if len(text) <= self.chunk_size:
            return [text] if text else []

        paragraphs = self._split_by_paragraphs(text)
        chunks: List[str] = []
        current_chunk: str = ""

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
                        temp_chunk: str = ""
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

        return [c for c in chunks if c.strip()]
