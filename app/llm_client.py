"""LLM chat client for generating RAG answers.

Sends retrieved context chunks along with the user's question to an
OpenAI-compatible chat-completions endpoint.  When ``OPENAI_API_KEY``
is ``dummy-key``, a mock answer is returned instead, allowing full
end-to-end testing without a real LLM.
"""

import httpx
from typing import List, Dict, Optional
from .config import settings
from .schemas import SearchResult


class LLMClient:
    """Generate answers from an OpenAI-compatible chat model.

    Attributes:
        api_key: OpenAI API key (``dummy-key`` enables mock mode).
        base_url: Base URL of the chat completions endpoint.
        model: Model identifier sent to the API.
    """

    def __init__(self) -> None:
        self.api_key: str = settings.OPENAI_API_KEY
        self.base_url: str = settings.OPENAI_BASE_URL.rstrip("/")
        self.model: str = settings.CHAT_MODEL

    async def generate_answer(
        self,
        query: str,
        context_chunks: List[SearchResult],
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """Generate an answer grounded in *context_chunks*.

        The system prompt instructs the model to answer *only* from the
        provided context and to explicitly say so when the context lacks
        the answer.

        Args:
            query: User's natural-language question.
            context_chunks: Retrieved chunks to include as context.
            conversation_history: Optional OpenAI-style message list for
                multi-turn dialogue (inserted between system and user messages).

        Returns:
            The LLM's answer as a plain string.
        """
        if self.api_key == "dummy-key":
            return self._mock_answer(query, context_chunks)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context_chunks)

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_prompt})

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1000,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

        return result["choices"][0]["message"]["content"]

    @staticmethod
    def _build_system_prompt() -> str:
        """Return the system prompt that constrains the LLM to context-only answers."""
        return """你是一个专业的知识库问答助手。请严格基于提供的上下文信息回答用户的问题。

规则：
1. 只能使用提供的上下文中的信息来回答问题
2. 如果上下文中没有相关信息，请明确说明"根据提供的文档内容，无法回答该问题"
3. 回答要准确、简洁、有条理
4. 如果上下文中存在多个相关信息点，请整合后回答
5. 不要编造或推断上下文中没有的信息"""

    @staticmethod
    def _build_user_prompt(query: str, context_chunks: List[SearchResult]) -> str:
        """Assemble the user-facing prompt with context and question.

        Each chunk is annotated with its source filename and similarity
        score so the LLM can weigh more relevant chunks higher.
        """
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[文档片段 {i} (来源: {chunk.filename}, 相似度: {chunk.similarity_score:.3f})]\n"
                f"{chunk.content}\n"
            )

        context_str = "\n".join(context_parts)

        return f"""请根据以下上下文信息回答用户问题。

上下文信息：
{context_str}

用户问题：{query}

请基于上述上下文信息回答问题。如果上下文中没有相关信息，请明确说明。"""

    def _mock_answer(self, query: str, context_chunks: List[SearchResult]) -> str:
        """Return a mock answer when no real LLM API key is configured.

        Concatenates the top-2 chunks (truncated to 300 chars) and wraps
        them in a template answer, with a note that this is simulated.
        """
        if not context_chunks:
            return "根据提供的文档内容，无法回答该问题。当前知识库中没有相关文档。"

        relevant_contents = [c.content for c in context_chunks[:2]]
        combined = " ".join(relevant_contents)

        if len(combined) > 300:
            combined = combined[:300] + "..."

        return (
            f"根据文档内容，关于\"{query}\"的相关信息如下：\n\n"
            f"{combined}\n\n"
            f"（注：这是模拟回答，未连接真实 LLM 服务。配置正确的 API Key 后将获得真实回答。）"
        )
