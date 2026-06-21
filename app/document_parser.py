"""Document parsing for PDF and TXT uploads.

Provides a uniform :meth:`DocumentParser.parse` interface that returns
the extracted plain text and the detected file type.  Encoding detection
for TXT files tries UTF-8 first, then GBK (common for Chinese text),
and finally falls back to UTF-8 with replacement characters.
"""

import io
from typing import Tuple
from pypdf import PdfReader


class DocumentParser:
    """Stateless parser that converts uploaded files into plain text."""

    @staticmethod
    def parse_txt(content: bytes) -> str:
        """Decode a TXT file from raw bytes to a Unicode string.

        Tries UTF-8 → GBK → UTF-8-with-replace to handle files saved
        with different encodings.

        Args:
            content: Raw file bytes.

        Returns:
            Decoded text with leading/trailing whitespace stripped.
        """
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("gbk")
            except UnicodeDecodeError:
                text = content.decode("utf-8", errors="ignore")
        return text.strip()

    @staticmethod
    def parse_pdf(content: bytes) -> str:
        """Extract text from a PDF file using :mod:`pypdf`.

        Each page's text is extracted independently and joined with
        newlines.  Scanned PDFs (image-only) will produce empty strings.

        Args:
            content: Raw PDF file bytes.

        Returns:
            Extracted text with leading/trailing whitespace stripped.
        """
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text())
        return "\n".join(text_parts).strip()

    @classmethod
    def parse(cls, filename: str, content: bytes) -> Tuple[str, str]:
        """Dispatch to the correct parser based on file extension.

        Args:
            filename: Original upload filename (used only for extension detection).
            content: Raw file bytes.

        Returns:
            ``(text, file_type)`` where *file_type* is ``'txt'`` or ``'pdf'``.

        Raises:
            ValueError: If the file extension is not ``.txt`` or ``.pdf``.
        """
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext == "txt":
            return cls.parse_txt(content), "txt"
        elif ext == "pdf":
            return cls.parse_pdf(content), "pdf"
        else:
            raise ValueError(f"Unsupported file format: .{ext}")
