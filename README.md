# RAG 个人知识库问答 API

一个基于检索增强生成（RAG）技术的个人知识库问答 RESTful API，支持文档上传、自动分块、向量化存储和智能问答。

## 功能特性

- 📄 **文档上传**：支持 PDF/TXT 文档上传与解析
- ✂️ **智能分块**：自定义分块算法，支持段落、句子级别的分块策略
- 🔍 **向量检索**：基于余弦相似度的语义检索引擎
- 🤖 **智能问答**：结合检索上下文的 LLM 问答
- 💾 **本地存储**：本地 JSON 持久化存储
- 🚫 **文件校验**：自动拒绝非文本格式文件上传

## 技术栈

- **Web 框架**：FastAPI
- **向量计算**：NumPy + scikit-learn
- **PDF 解析**：pypdf
- **LLM 集成**：OpenAI 兼容 API

## 项目结构

```
wyl-0004/
├── app/
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── schemas.py         # Pydantic 数据模型
│   ├── document_parser.py # 文档解析器
│   ├── chunker.py         # 文本分块器
│   ├── embedder.py        # 向量化模块
│   ├── vector_store.py    # 向量存储与检索
│   ├── llm_client.py     # LLM 客户端
│   └── main.py           # FastAPI 主应用
├── sample_data/
│   └── test_document.txt # 测试文档
├── data/                   # 向量数据存储目录（自动创建）
├── requirements.txt
├── .env.example
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置你的 API Key：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=3
VECTOR_STORE_PATH=./data/vector_store.json
```

> **注意**：如果不配置 API Key，系统将使用模拟模式运行，可以测试功能，但回答将是模拟的。

### 3. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后，访问：
- API 文档（Swagger UI）：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## API 文档

### 1. 健康检查

**GET** `/health`

检查服务是否正常运行。

### 2. 文档上传

**POST** `/api/documents/upload`

上传 PDF 或 TXT 文档，系统将自动解析、分块并向量化存储。

**请求参数**：
- `file` (form-data): 要上传的文件（.txt 或 .pdf）

**响应示例**：
```json
{
  "success": true,
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "test_document.txt",
  "chunks_count": 5,
  "message": "Document uploaded successfully. Created 5 chunks."
}
```

### 3. 列出文档

**GET** `/api/documents`

获取所有已上传文档列表。

**响应示例**：
```json
{
  "success": true,
  "documents": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "test_document.txt",
      "file_type": "txt",
      "chunks_count": 5
    }
  ],
  "total": 1
}
```

### 4. 删除文档

**DELETE** `/api/documents/{doc_id}`

删除指定文档及其向量数据。

### 5. 向量检索

**POST** `/api/search`

根据查询语义检索相关文档片段。

**请求体**：
```json
{
  "query": "什么是RAG技术？",
  "top_k": 3
}
```

**响应示例**：
```json
{
  "success": true,
  "query": "什么是RAG技术？",
  "results": [
    {
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000_2",
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "test_document.txt",
      "content": "检索增强生成（Retrieval-Augmented Generation，简称RAG）是一种将信息检索与文本生成相结合的AI技术框架...",
      "similarity_score": 0.895
    }
  ],
  "total": 1
}
```

### 6. 对话式问答

**POST** `/api/chat`

基于知识库内容进行智能问答。

**请求体**：
```json
{
  "query": "RAG相比纯大模型问答有什么优势？",
  "top_k": 3,
  "conversation_history": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
  ]
}
```

**响应示例**：
```json
{
  "success": true,
  "query": "RAG相比纯大模型问答有什么优势？",
  "answer": "RAG相比纯大模型问答具有以下优势：1. 准确性更高：回答基于真实的文档内容，减少幻觉；2. 可追溯性：可以查看回答的来源文档；3. 成本更低：无需微调模型即可更新知识；4. 实时性：可以随时添加新的知识库内容。",
  "sources": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "filename": "test_document.txt",
      "content": "RAG相比纯大模型问答具有以下优势：...",
      "similarity_score": 0.92
    }
  ],
  "model": "gpt-4o-mini"
}
```

## curl 示例

### 1. 上传 TXT 文档

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_data/test_document.txt"
```

### 2. 向量检索

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是RAG技术？", "top_k": 3}'
```

### 3. 智能问答

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "RAG有哪些优势？"}'
```

### 4. 验证非文本文件拒绝

```bash
# 创建一个测试图片
echo "not an image" > test.jpg

curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.jpg"
```

**预期响应（HTTP 400）：
```json
{
  "success": false,
  "error": 400,
  "message": "Unsupported file format. Allowed formats: .txt, .pdf"
}
```

### 5. 查看已上传文档

```bash
curl "http://localhost:8000/api/documents"
```

### 6. 健康检查

```bash
curl "http://localhost:8000/health"
```

## 验证步骤

按照以下步骤验证系统功能：

1. **启动服务**：
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. **上传 TXT 文档**：
   ```bash
   curl -X POST "http://localhost:8000/api/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_data/test_document.txt"
   ```

3. **针对文档内容提问**：
   ```bash
   curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "RAG技术包含哪三个核心步骤？"}'
   ```

4. **验证回答内容与文档相符且输出格式为 JSON**：
   - 检查 `success` 字段为 `true`
   - 检查 `answer` 字段包含文档中的三个步骤
   - 检查 `sources` 字段包含相关文档片段

5. **验证非文本文件上传拒绝**：
   ```bash
   curl -X POST "http://localhost:8000/api/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test.jpg"
   ```
   确认返回 HTTP 400 错误，提示不支持的文件格式。

## 核心模块说明

### 分块算法（chunker.py）

自定义分块逻辑，不依赖任何 RAG 框架：
- 按段落分割，再按句子细粒度分块
- 支持配置分块大小（CHUNK_SIZE）和重叠大小（CHUNK_OVERLAP）
- 长句子自动按词分割
- 块间重叠确保上下文连续性

### 向量检索（vector_store.py）

手写向量检索引擎：
- 基于余弦相似度计算
- 支持 Top-K 检索
- 本地 JSON 持久化存储
- 自动重建向量索引

## 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| CHUNK_SIZE | 分块字符数 | 500 |
| CHUNK_OVERLAP | 分块重叠字符数 | 100 |
| TOP_K | 检索返回数量 | 3 |
| MAX_FILE_SIZE | 最大文件大小 | 10MB |

## 注意事项

1. 本项目**不使用**任何现成的 RAG 框架（如 LlamaIndex、Haystack），所有核心逻辑均为手写实现
2. 支持的文件格式：`.txt`、`.pdf`
3. 最大文件大小：10MB
4. 向量数据存储在本地 `data/vector_store.json` 文件中
