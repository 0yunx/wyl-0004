import httpx
from typing import List, Dict, Optional
from .config import settings
from .schemas import SearchResult


class LLMClient:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.base_url = settings.OPENAI_BASE_URL.rstrip("/")
        self.model = settings.CHAT_MODEL

    async def generate_answer(
        self,
        query: str,
        context_chunks: List[SearchResult],
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        if self.api_key == "dummy-key":
            return self._mock_answer(query, context_chunks)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context_chunks)

        messages = [
            {"role": "system", "content": system_prompt},
        ]

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
        return """你是一个专业的知识库问答助手。请严格基于提供的上下文信息回答用户的问题。

规则：
1. 只能使用提供的上下文中的信息来回答问题
2. 如果上下文中没有相关信息，请明确说明"根据提供的文档内容，无法回答该问题"
3. 回答要准确、简洁、有条理
4. 如果上下文中存在多个相关信息点，请整合后回答
5. 不要编造或推断上下文中没有的信息"""

    @staticmethod
    def _build_user_prompt(query: str, context_chunks: List[SearchResult]) -> str:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"[文档片段 {i} (来源: {chunk.filename}, 相似度: {chunk.similarity_score:.3f})]\n{chunk.content}\n")

        context_str = "\n".join(context_parts)

        return f"""请根据以下上下文信息回答用户问题。

上下文信息：
{context_str}

用户问题：{query}

请基于上述上下文信息回答问题。如果上下文中没有相关信息，请明确说明。"""

    def _mock_answer(self, query: str, context_chunks: List[SearchResult]) -> str:
        if not context_chunks:
            return "根据提供的文档内容，无法回答该问题。当前知识库中没有相关文档。"

        relevant_contents = [c.content for c in context_chunks[:2]]
        combined = " ".join(relevant_contents)

        if len(combined) > 300:
            combined = combined[:300] + "..."

        answer = f"根据文档内容，关于\"{query}\"的相关信息如下：\n\n{combined}\n\n（注：这是模拟回答，未连接真实 LLM 服务。配置正确的 API Key 后将获得真实回答。）"
        return answer
