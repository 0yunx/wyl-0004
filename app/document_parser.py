import io
from typing import Tuple
from pypdf import PdfReader


class DocumentParser:
    @staticmethod
    def parse_txt(content: bytes) -> str:
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
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text())
        return "\n".join(text_parts).strip()

    @classmethod
    def parse(cls, filename: str, content: bytes) -> Tuple[str, str]:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext == "txt":
            return cls.parse_txt(content), "txt"
        elif ext == "pdf":
            return cls.parse_pdf(content), "pdf"
        else:
            raise ValueError(f"Unsupported file format: .{ext}")
