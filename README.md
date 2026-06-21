# RAG 个人知识库问答 API v2.0

一个基于检索增强生成（RAG）技术的个人知识库问答 RESTful API，支持文档上传、自动分块、向量化存储和智能问答。

## 功能特性

- 📄 **文档上传**：支持 PDF/TXT 文档上传与解析
- ✂️ **智能分块**：自定义分块算法，支持段落、句子级别的分块策略
- 🔍 **向量检索**：基于 HNSW 索引的近似最近邻检索，O(log N) 查询复杂度
- 🤖 **智能问答**：结合检索上下文的 LLM 问答
- 💾 **高效存储**：二进制向量文件 + 内存映射加载，避免 JSON 序列化膨胀
- ♻️ **自动回收**：软删除 + 垃圾比例超阈值自动压实
- 🚫 **文件校验**：自动拒绝非文本格式文件上传
- 🔒 **原子写入**：元数据和索引文件通过 tmp+rename 保证崩溃安全
- 🧠 **内容去重缓存**：基于 SHA-256 全文哈希的 SQLite 缓存，相同文档跳过向量化，节省 tokens 和 API 额度

## 技术栈

- **Web 框架**：FastAPI + Pydantic v2
- **向量索引**：手写 HNSW（Hierarchical Navigable Small World）
- **向量计算**：NumPy（内存映射 + L2 归一化 + 点积余弦）
- **PDF 解析**：pypdf
- **LLM 集成**：OpenAI 兼容 API（支持模拟模式回退）

## 内容去重缓存机制

为避免重复上传相同文档时浪费 Embedding API 的 tokens 和额度，系统实现了基于 **SHA-256 全文哈希 + SQLite 持久化缓存** 的两级去重机制。

### 工作流程

1. **哈希计算**：文档解析为纯文本后，对全文 UTF-8 编码计算 SHA-256 十六进制摘要作为唯一标识
2. **两级查询**：
   - 首先通过 `content_hash` 在 `VectorStore` 中查找是否已有文档引用相同物理向量行（`find_row_indices_by_hash`）
   - 命中则直接复用已有 row indices，不重复写入 `vectors.bin`，也不重复插入 HNSW 索引
   - 未命中则查询 SQLite 缓存表 (`dedupe_cache`) 获取缓存的 chunks 文本和向量
3. **未命中处理**：两级缓存都未命中时，正常执行 分块 → 向量化 链路，并将结果写入 SQLite 缓存
4. **引用计数**：`VectorStore` 内部维护 `_row_refcount`，物理向量行被多个文档共享时引用计数递增；删除文档时仅当计数归零才真正软删除该行

### SQLite 缓存表结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `hash` | TEXT (PRIMARY KEY) | SHA-256 十六进制摘要 |
| `chunks_json` | TEXT NOT NULL | JSON 数组，分块后的文本 |
| `vectors_json` | TEXT NOT NULL | JSON 二维数组，向量数据 |
| `vector_dim` | INTEGER NOT NULL | 向量维度 |
| `chunk_count` | INTEGER NOT NULL | 分块数量 |
| `created_at` | TEXT NOT NULL | ISO-8601 首次缓存时间 |

### API 响应变化

`POST /api/documents/upload` 响应新增字段：
- `deduplicated` (bool)：`true` 表示本次命中缓存并复用了向量化结果，`false` 表示首次上传走完整链路

## v2 核心改进

| 问题 | v1 | v2 |
|------|----|----|
| 向量存储 | JSON 序列化 float32 列表（~20 bytes/float） | 二进制 `.bin` 文件（4 bytes/float），5 倍压缩 |
| 内存占用 | 启动时全量 `np.array()` 加载 | `np.memmap` 按需映射，1GB 存储不占 1GB RAM |
| 写入放大 | 每次上传全量 JSON 序列化 | 追加写入二进制向量，仅重写元数据和索引 |
| 检索性能 | O(N) 暴力搜索 | O(log N) HNSW 近似最近邻，带回退暴力兜底 |
| 数据安全 | 崩溃时可能损坏 | 原子写入（tmp + os.replace） |
| 垃圾回收 | 无 | 软删除 + 垃圾占比 >20% 自动压实 |

## 项目结构

```
wyl-0004/
├── app/
│   ├── __init__.py
│   ├── config.py          # 配置管理（含 HNSW 参数）
│   ├── schemas.py         # Pydantic v2 数据模型
│   ├── document_parser.py # 文档解析器
│   ├── chunker.py         # 文本分块器
│   ├── embedder.py        # 向量化模块
│   ├── cache.py           # 内容去重缓存（SQLite + SHA-256）
│   ├── vector_store.py    # 向量存储与检索（二进制 + memmap + HNSW）
│   ├── hnsw.py            # HNSW 索引实现
│   ├── llm_client.py      # LLM 客户端
│   └── main.py            # FastAPI 主应用
├── sample_data/
│   └── test_document.txt  # 测试文档
├── data/                   # 向量数据存储目录（自动创建）
│   ├── metadata.json       # 文档和分块元数据
│   ├── vectors.bin         # 原始 float32 二进制向量
│   ├── hnsw_index.json     # HNSW 索引快照
│   └── dedupe_cache.db     # SQLite 去重缓存数据库
├── requirements.txt
├── .env.example
├── .gitignore
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
DATA_DIR=./data
HNSW_M=16
HNSW_EF_CONSTRUCTION=200
HNSW_EF=50
CACHE_DB_PATH=./data/dedupe_cache.db
```

> **注意**：如果不配置 API Key（保持默认 `dummy-key`），系统将使用模拟模式运行，可以测试功能，但回答将是模拟的。

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

**响应示例（首次上传）**：
```json
{
  "success": true,
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "test_document.txt",
  "chunks_count": 5,
  "deduplicated": false,
  "message": "Document uploaded successfully. Created 5 chunks."
}
```

**响应示例（重复上传命中缓存）**：
```json
{
  "success": true,
  "document_id": "a1b2c3d4-e29b-41d4-a716-446655440000",
  "filename": "test_document.txt",
  "chunks_count": 5,
  "deduplicated": true,
  "message": "Document uploaded successfully. Reused 5 cached chunks (deduplicated)."
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

软删除指定文档及其向量数据。当垃圾数据超过总数据 20% 时自动触发压实（compaction），回收磁盘空间并重建 HNSW 索引。

### 5. 向量检索

**POST** `/api/search`

根据查询语义检索相关文档片段。使用 HNSW 索引进行近似最近邻搜索，对归一化向量用余弦相似度重排序。

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

## curl 示例

### 1. 上传 TXT 文档

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
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
echo "not an image" > test.jpg
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@test.jpg"
```

**预期响应（HTTP 400）**：
```json
{
  "success": false,
  "error": 400,
  "message": "Unsupported file format. Allowed formats: .pdf, .txt"
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
     -F "file=@sample_data/test_document.txt"
   ```
   预期：`deduplicated=false`，返回 `chunks_count` 和新的 `document_id`。

3. **再次上传完全相同的文档**：
   ```bash
   curl -X POST "http://localhost:8000/api/documents/upload" \
     -F "file=@sample_data/test_document.txt"
   ```
   预期：`deduplicated=true`，`chunks_count` 与第一次一致，但 `document_id` 不同（新文档记录指向同一组物理向量）。

4. **检查 SQLite 缓存记录**：
   ```bash
   sqlite3 data/dedupe_cache.db "SELECT hash, chunk_count, created_at FROM dedupe_cache;"
   ```
   预期：确有一条缓存记录。

5. **针对文档内容提问**：
   ```bash
   curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "RAG技术包含哪三个核心步骤？"}'
   ```

6. **验证回答内容与文档相符且输出格式为 JSON**：
   - 检查 `success` 字段为 `true`
   - 检查 `answer` 字段包含文档中的三个步骤
   - 检查 `sources` 字段包含相关文档片段

7. **验证非文本文件上传拒绝**：
   ```bash
   echo "not an image" > test.jpg
   curl -X POST "http://localhost:8000/api/documents/upload" \
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

### 去重缓存（cache.py）

SQLite + SHA-256 内容去重：
- 标准库 `sqlite3`，零新增依赖
- 对全文文本计算 SHA-256 哈希作为缓存键
- 两级命中：先查 `VectorStore` 的内存索引（复用已有物理向量），再查 SQLite（复用已缓存的向量化结果）
- 引用计数保证共享向量的安全删除

### HNSW 索引（hnsw.py）

手写 HNSW 近似最近邻索引，基于原始论文算法：
- 多层跳表结构，O(log N) 搜索复杂度
- 启发式邻居选择（Algorithm 4），保证图连通性
- 软删除支持：修补邻居边，保持图结构完整
- 参数：M=16, M_max0=32, ef_construction=200, ef=50

### 向量存储（vector_store.py）

高效向量存储引擎：
- **二进制存储**：`vectors.bin` 存储 raw float32（4 bytes/dim），对比 JSON 列表的 ~20 bytes/dim
- **内存映射**：`np.memmap` 只读映射，1GB 向量文件不占 1GB 常驻内存
- **追加写入**：新向量以 `"ab"` 模式追加，不重写整个文件
- **L2 归一化**：存储前归一化，使欧氏距离与余弦距离单调相关：cos(q,v) = 1 - ||q-v||²/2
- **双路检索**：HNSW 主路径（3x 过采样 + 余弦重排）+ 暴力兜底
- **原子写入**：metadata 和 index 文件通过 tmp + os.replace 保证崩溃安全
- **自动压实**：软删除后垃圾占比 >20% 触发 compact，回收空间并重建索引
- **引用计数**：多个文档可共享同一组物理向量，删除时按引用计数回收

## 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| CHUNK_SIZE | 分块字符数 | 500 |
| CHUNK_OVERLAP | 分块重叠字符数 | 100 |
| TOP_K | 检索返回数量 | 3 |
| MAX_FILE_SIZE | 最大文件大小 | 10MB |
| DATA_DIR | 数据存储目录 | ./data |
| CACHE_DB_PATH | SQLite 去重缓存路径 | ./data/dedupe_cache.db |
| HNSW_M | HNSW 每层最大连接数 | 16 |
| HNSW_EF_CONSTRUCTION | HNSW 构建时搜索宽度 | 200 |
| HNSW_EF | HNSW 查询时搜索宽度 | 50 |

## 注意事项

1. 本项目**不使用**任何现成的 RAG 框架（如 LlamaIndex、Haystack），所有核心逻辑均为手写实现
2. 支持的文件格式：`.txt`、`.pdf`
3. 最大文件大小：10MB
4. 向量数据存储在 `data/` 目录下：`metadata.json`（元数据）、`vectors.bin`（二进制向量）、`hnsw_index.json`（HNSW 索引）、`dedupe_cache.db`（SQLite 去重缓存）
5. 删除文档为软删除，垃圾数据超过 20% 时自动压实回收空间
6. 去重缓存使用标准库 `sqlite3`，无需额外 pip 依赖
