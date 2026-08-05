# Cayz-Agent

具备持久化记忆、联网搜索、知识库检索（RAG）、多模型支持与多 Agent 协作的企业级 AI Agent。提供 Web 聊天界面和 REST API。

## 架构概览

```
┌──────────────────────────────────────────────────┐
│              Web 聊天界面（web_app.html）           │
│  会话管理 · 流式对话 · 知识库管理 · 工具权限控制     │
└──────────────────────┬───────────────────────────┘
                       │ HTTP/SSE
┌──────────────────────▼───────────────────────────┐
│                FastAPI REST API                   │
│  鉴权 · 限流 · CORS · 请求校验 · 监控             │
│  /chat  /chat/stream  /sessions  /knowledge       │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│              LangGraph Agent 引擎                  │
│  ┌──────────┐  ┌──────────────────────────────┐  │
│  │ 单 Agent  │  │      多 Agent 协作            │  │
│  │ 决策节点  │  │ Router → Knowledge/Search/    │  │
│  │ + 工具调用 │  │          Chat/Business Agent  │  │
│  └──────────┘  └──────────────────────────────┘  │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│                   基础设施层                        │
│  LLM · ChromaDB · SQLite · Tavily · CRM · SMTP    │
└──────────────────────────────────────────────────┘
```

## 核心功能

- **多模型支持**：OpenAI / 通义千问 / 智谱 GLM / 百度文心 / Ollama 本地推理
- **持久化记忆**：SQLite 会话检查点，支持跨会话上下文保持
- **联网搜索**：基于 Tavily API 的实时互联网搜索
- **知识库检索（RAG）**：ChromaDB 向量存储 + 文档切片 + 语义检索，支持 .txt / .md / .pdf
- **多 Agent 协作**：路由 Agent 自动分发到知识库 / 搜索 / 通用对话 / 业务集成子 Agent
- **业务系统集成**：CRM 客户查询、企业微信通知、SMTP 邮件发送
- **工具扩展**：Excel 文件解析、二维码生成等
- **企业级安全**：API Key 鉴权（三级权限）、请求限流、输入校验、敏感内容检测、HTTPS 强制
- **可观测性**：Prometheus 指标导出、结构化日志、告警 watcher

## 项目结构

```
cayz-agent/
├── cayz_agent/              # 核心包
│   ├── __init__.py          # 公开 API
│   ├── __main__.py          # CLI 入口
│   ├── api.py               # FastAPI REST 服务（21+ 端点）
│   ├── graph.py             # 单 Agent LangGraph 图
│   ├── multi_agent.py       # 多 Agent 协作架构
│   ├── tools.py             # 工具集（25+ 工具）
│   ├── rag.py               # RAG 子系统（ChromaDB）
│   ├── llm.py               # 多模型工厂
│   ├── config.py            # 集中配置管理
│   ├── session.py           # 会话管理
│   ├── middleware.py         # 鉴权 + 限流中间件
│   ├── monitor.py           # Prometheus 指标
│   ├── alerts.py            # 后台告警 watcher
│   ├── cache.py             # 多层缓存（LLM / Embedding / RAG）
│   ├── retry.py             # 指数退避重试
│   ├── validators.py        # 输入校验
│   ├── sanitizers.py        # 内容安全审查
│   ├── exceptions.py        # 异常层次结构
│   ├── app_state.py         # 应用状态管理
│   ├── request_context.py   # 请求上下文
│   └── integrations/        # 业务系统集成
│       ├── crm.py           # CRM 客户系统
│       ├── notify.py        # 企业微信通知
│       └── email_sender.py  # SMTP 邮件
├── web_app.html             # Web 聊天界面（单文件前端）
├── tests/                   # 测试套件（828 tests）
├── nginx/                   # Nginx 反向代理配置
│   ├── nginx.conf           # TLS 生产模式配置
│   └── frontend.conf        # 前端静态文件服务配置
├── scripts/                 # 运维脚本（SQLite 备份）
├── benchmarks/              # 性能基准测试
├── Dockerfile               # 多阶段构建（安全加固）
├── docker-compose.yml       # 编排（direct / tls 双模式）
├── pyproject.toml           # 项目元数据
├── requirements.lock        # 锁定依赖（含哈希）
└── .env.example             # 环境变量模板
```

## 快速开始

### 前置条件

- Python >= 3.12
- 配置 `.env` 文件（至少需要 `OPENAI_API_KEY` 和 `TAVILY_API_KEY`）

### 方式一：Docker（推荐）

```bash
# 复制环境变量模板
cp .env.example .env
# 编辑 .env，填入真实的 API Key

# 开发/内网模式（HTTP）
# 启动 API 服务（端口 8000）+ 前端界面（端口 3000）
docker compose up -d

# 生产模式（TLS + Nginx 反向代理）
docker compose --profile tls up -d
```

- API 文档：http://localhost:8000/docs（需设置 `DOCS_ENABLED=true`）
- Web 聊天界面：http://localhost:3000

### 方式二：本地 Python 虚拟环境

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装依赖（清华镜像源）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.lock

# 安装项目
pip install --no-deps -e .

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入真实的 API Key

# 启动 API 服务
python -m cayz_agent.api

# 另开终端，启动前端（Python 内置 HTTP 服务器）
python -m http.server 3000 --directory .
# 访问 http://localhost:3000/web_app.html
```

### CLI 交互模式

```bash
cayz-agent
# 或
python -m cayz_agent
```

## 配置说明

核心配置项（详见 `.env.example`）：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_PROVIDER` | 模型提供商：openai / zhipu / qwen / ernie / ollama | openai |
| `OPENAI_API_KEY` | OpenAI API Key（DashScope 兼容） | - |
| `TAVILY_API_KEY` | Tavily 联网搜索 API Key | - |
| `CHECKPOINT_BACKEND` | 持久化后端：memory / sqlite | memory |
| `API_KEY` | API 鉴权密钥（管理员） | - |
| `AUTH_REQUIRED` | 是否强制鉴权 | true |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 深度健康检查（含依赖状态） |
| `GET` | `/metrics` | Prometheus 指标导出 |
| `POST` | `/chat` | 同步对话 |
| `POST` | `/chat/stream` | 流式对话（SSE） |
| `POST` | `/chat/multi` | 多 Agent 对话 |
| `GET` | `/sessions` | 会话列表 |
| `GET` | `/sessions/{id}` | 会话详情 |
| `GET` | `/sessions/{id}/messages` | 会话消息历史 |
| `DELETE` | `/sessions/{id}` | 删除会话 |
| `GET` | `/knowledge/count` | 知识库文档片段数 |
| `GET` | `/knowledge/sources` | 知识库文档来源列表 |
| `GET` | `/knowledge/{source}` | 获取指定来源文档内容 |
| `POST` | `/knowledge/upload` | 上传知识库文档 |
| `POST` | `/knowledge/upload-file` | 上传文件到知识库 |
| `POST` | `/knowledge/batch-upload` | 批量上传文档 |
| `PUT` | `/knowledge/update` | 更新知识库文档 |
| `DELETE` | `/knowledge/{source}` | 删除知识库文档 |
| `GET` | `/knowledge/search` | 检索知识库 |
| `GET` | `/models` | 支持的模型列表 |
| `GET` | `/tools` | 可用工具列表 |
| `POST` | `/tools/call` | 直接调用工具 |

## 多模型支持

通过 `LLM_PROVIDER` 切换模型提供商：

| Provider | 说明 | 所需配置 |
|----------|------|----------|
| `openai` | OpenAI / DashScope 兼容 | `OPENAI_API_KEY` + `OPENAI_API_BASE` |
| `qwen` | 通义千问 | `OPENAI_API_KEY`（DashScope） |
| `zhipu` | 智谱 GLM | `ZHIPU_API_KEY` |
| `ernie` | 百度文心 | `ERNIE_API_KEY` + `ERNIE_SECRET_KEY` |
| `ollama` | 本地推理 | `OLLAMA_BASE_URL` |

## 多 Agent 协作

```
User → Router Agent → ┬→ Knowledge Agent（RAG 检索）
                       ├→ Search Agent（联网搜索）
                       ├→ Chat Agent（通用对话）
                       └→ Business Agent（CRM/通知/邮件）
```

路由 Agent 自动分析用户意图，分发到对应的专业子 Agent。通过 `/chat/multi` 端点或 Web UI 的"多 Agent"模式使用。

## 工具集

| 工具 | 功能 | 权限级别 |
|------|------|----------|
| `get_current_time` | 获取当前日期时间 | 只读 |
| `web_search` | 联网搜索 | 只读 |
| `knowledge_search` | 知识库检索 | 只读 |
| `knowledge_upload` | 上传知识库文档 | 读写 |
| `crm_query_customer` | CRM 客户查询 | 只读 |
| `crm_search_customers` | CRM 客户搜索 | 只读 |
| `crm_query_orders` | CRM 订单查询 | 只读 |
| `send_wecom_notify` | 企业微信通知 | 读写 |
| `send_email` | SMTP 邮件发送 | 读写 |
| `parse_excel` | Excel 文件解析 | 只读 |
| `generate_qrcode` | 二维码生成 | 只读 |

## 企业级特性

### 安全

- **三级 API Key 鉴权**：管理员 / 读写 / 只读
- **请求限流**：滑动窗口限流，读/写分离
- **输入校验**：所有用户输入和工具参数经过校验
- **敏感内容检测**：知识库上传敏感信息扫描
- **Prompt Injection 防护**：外部内容用 `<untrusted_content>` 标签包裹
- **HTTPS 强制**：生产模式自动 HTTP→HTTPS 重定向
- **请求体大小限制**：防 DoS
- **非 root 运行**：Docker 容器以 appuser 运行

### 可观测性

- **Prometheus 指标**：`/metrics` 端点导出请求量、token 使用、工具调用、延迟等
- **结构化日志**：支持 JSON 格式，便于接入 ELK / Loki / Datadog
- **告警 watcher**：后台周期性扫描指标，触发阈值时回调

### 缓存

- LLM 路由缓存（TTL + LRU）
- Embedding 向量缓存
- RAG 检索结果缓存

## 测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest tests/ -v

# 带覆盖率
pytest tests/ -v --cov=cayz_agent --cov-report=term-missing

# 覆盖率要求 >= 80%
pytest tests/ -v --cov=cayz_agent --cov-fail-under=80
```

## CI/CD

GitHub Actions 工作流（`.github/workflows/ci.yml`）：

1. **test**：Python 3.12 / 3.13 矩阵测试 + 覆盖率
2. **lint**：Ruff 代码检查 + 格式化验证
3. **security**：pip-audit 依赖漏洞扫描 + Trivy 文件系统扫描
4. **docker**：构建并推送镜像到 GHCR + Trivy 镜像漏洞扫描

安全扫描要求 0 个 HIGH/CRITICAL 漏洞。

## 许可证

详见 [LICENSE](LICENSE) 文件。