# cayz-agent

![CI](https://github.com/Cayz22/Cayz-agent/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![Tests](https://img.shields.io/badge/tests-772-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)

> 基于 LangGraph 的 AI Agent，具备持久化记忆、联网搜索、知识库检索（RAG）、多模型支持、多 Agent 协作、监控告警与业务系统集成能力。

## 项目简介

cayz-agent 是一个 AI Agent 项目。项目展示了完整的 Agent 工程化能力：持多模型、RAG 知识库、多 Agent 协作、监控告警、业务系统集成的综合智能体平台。

### 核心能力

| 能力 | 实现方式 | 对应模块 |
|------|---------|---------|
| 多轮对话 + 持久化记忆 | LangGraph StateGraph + MemorySaver/SqliteSaver | `graph.py` |
| 联网搜索 | Tavily API + 重试机制 | `tools.py` |
| 知识库检索（RAG） | ChromaDB + OpenAI Embeddings | `rag.py` |
| 多模型支持 | OpenAI / 智谱 GLM / 通义千问 / 百度文心 / Ollama | `llm.py` |
| 多 Agent 协作 | 路由 Agent + 4 个专业子 Agent（知识/搜索/通用/业务集成） | `multi_agent.py` |
| 监控与告警 | Prometheus 指标 + 阈值告警 + X-Request-ID 追踪 | `monitor.py` / `alerts.py` / `request_context.py` |
| 业务系统集成 | CRM + 企业微信 + SMTP 邮件 | `integrations/` |
| REST API 服务 | FastAPI + SSE 流式输出 + 鉴权限流 + CORS + 优雅停机 | `api.py` / `middleware.py` / `app_state.py` |
| 会话管理 | 列表/查询/删除会话，基于 SQLite | `session.py` |
| 知识库管理 | 文档删除/更新/批量导入/来源查询 | `rag.py` |
| 结构化日志 | text / json 可切换，含 request_id，便于接入 ELK/Loki | `config.py` |
| 异常体系 | 分层自定义异常（LLM/Tool/RAG/Integration） | `exceptions.py` |
| Web 界面 | Streamlit（白底金线高雅主题） | `web_app.py` |
| 安全护栏 | 输入验证 + 输出脱敏 + 提示词注入检测 + API Key 鉴权 + 限流 + 文件路径白名单 | `validators.py` / `sanitizers.py` / `middleware.py` |
| Agent 工具集 | 25 个工具，按风险三级权限分级（readonly/write/admin） | `tools.py` |

## 架构设计

### 单 Agent 架构（`graph.py`）

```
用户输入 → validate_input → agent（LLM + 工具）→ tools → agent → ... → END
```

### 多 Agent 协作架构（`multi_agent.py`）

```
                    ┌──────────────┐
                    │ Router Agent │ (意图识别)
                    └──────┬───────┘
                           │
       ┌──────────────┬────┴────────────┬──────────────┐
       ▼              ▼                 ▼              ▼
┌─────────────┐┌───────────┐ ┌───────────┐ ┌─────────────┐
│Knowledge Agt││Search Agt │ │ Chat Agent │ │Business Agt │
│  (RAG 检索)  ││(联网搜索)  │ │ (通用对话)  │ │(CRM/通知/邮件)│
└─────────────┘└───────────┘ └───────────┘ └─────────────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                           │
                    ┌──────▼──────┐
                    │    Tools    │ (工具执行)
                    └─────────────┘
```

路由 Agent 分析用户意图，分流到对应专业子 Agent：
- **知识库 Agent**：处理私有知识问答（RAG 检索）
- **搜索 Agent**：处理实时信息查询（联网搜索）
- **通用 Agent**：处理闲聊、问候、通用问答
- **业务集成 Agent**：处理 CRM 客户/订单查询、企业微信通知、邮件发送

## 技术栈

- **Agent 框架**：LangGraph 0.0.30+ / LangChain 0.1+
- **大模型接入**：LangChain OpenAI（兼容 5 个 provider）
- **向量数据库**：ChromaDB
- **Embeddings**：OpenAI text-embedding-3-small
- **Web 框架**：FastAPI + Uvicorn
- **前端**：Streamlit
- **配置管理**：pydantic-settings
- **重试机制**：tenacity
- **缓存层**：cachetools（TTL+LRU，覆盖 LLM/Embedding/RAG 检索）
- **容器化**：Docker + docker-compose
- **CI/CD**：GitHub Actions
- **测试**：pytest（772+ 测试用例，覆盖率 93%，含端到端集成测试）

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd cayz-agent

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate  # Windows PowerShell
# .venv\Scripts\activate.bat  # Windows CMD
# source .venv/bin/activate    # Linux/macOS

# 安装依赖（国内用户可加 -i https://pypi.tuna.tsinghua.edu.cn/simple 使用清华镜像源加速）
pip install -e .
```

### 2. 配置

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入 API Key
# 必填：OPENAI_API_KEY（或对应 provider 的 key）
# 可选：TAVILY_API_KEY（联网搜索）
# 生产建议：API_KEY（API 鉴权）、CORS_ALLOWED_ORIGINS
```

### 3. 多模型配置

在 `.env` 中切换 `LLM_PROVIDER`：

| Provider | 环境变量 | 示例 MODEL_NAME |
|----------|---------|----------------|
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `qwen` | `OPENAI_API_KEY`（DashScope） | `qwen-turbo` |
| `zhipu` | `ZHIPU_API_KEY` | `glm-4` |
| `ernie` | `ERNIE_API_KEY` | `ernie-bot-4` |
| `ollama` | 无需 key | `llama3` |

### 4. 启动方式

```bash
# 方式一：CLI 对话
python -m cayz_agent

# 方式二：Web 界面（Streamlit）
streamlit run web_app.py

# 方式三：REST API 服务
cayz-agent-api
# 或
python -m cayz_agent.api

# 方式四：使用多 Agent 架构（代码集成）
from cayz_agent import create_multi_agent_graph
app = create_multi_agent_graph()
```

### 5. API 调用示例

```bash
# 健康检查（深度依赖检查）
curl http://localhost:8000/health

# Prometheus 指标
curl http://localhost:8000/metrics

# 对话（启用 API_KEY 后需带 X-API-Key 或 Authorization: Bearer）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"message": "你好"}'

# 流式对话（SSE）
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "今天天气怎么样"}'

# 会话管理
curl http://localhost:8000/sessions                         # 列出所有会话
curl http://localhost:8000/sessions/{thread_id}             # 查询会话详情
curl -X DELETE http://localhost:8000/sessions/{thread_id}   # 删除会话

# 知识库管理
curl http://localhost:8000/knowledge/sources                # 列出文档来源
curl http://localhost:8000/knowledge/count                  # 文档片段总数
curl -X POST http://localhost:8000/knowledge/upload \
  -H "Content-Type: application/json" \
  -d '{"text": "要上传的知识", "source": "manual"}'
curl -X POST http://localhost:8000/knowledge/batch-upload \
  -H "Content-Type: application/json" \
  -d '{"items": [{"text": "doc1", "source": "s1"}, {"text": "doc2", "source": "s2"}]}'
curl -X PUT http://localhost:8000/knowledge/update \
  -H "Content-Type: application/json" \
  -d '{"source": "manual", "text": "更新后的内容"}'
curl -X DELETE http://localhost:8000/knowledge/{source}     # 按来源删除
```

## RAG 知识库使用

### 通过 Agent 工具上传

直接对话：
> "请记住以下知识：cayz-agent 是一个基于 LangGraph 的 AI Agent 项目..."

Agent 会自动调用 `knowledge_upload` 工具存入知识库。

### 通过代码操作

```python
from cayz_agent.rag import get_rag_manager

manager = get_rag_manager()

# 上传文本
manager.add_documents("这是要存入的知识文本", source="manual")

# 上传文件（支持 .txt / .md / .pdf）
manager.add_file("./docs/product_manual.pdf")

# 检索
results = manager.search("产品使用方法", top_k=3)
for doc in results:
    print(doc.page_content)
```

## 监控与告警

### Prometheus 指标

访问 `/metrics` 获取 Prometheus 格式指标：

```
cayz_agent_requests_total{type="chat",success="true"} 42
cayz_agent_token_usage_total{kind="input"} 15238
cayz_agent_tool_calls_total{name="web_search",success="true"} 7
cayz_agent_request_latency_seconds_bucket{le="0.5"} 38
```

### 告警规则

| 规则 | 阈值 | 级别 |
|------|------|------|
| 高错误率 | >50% | CRITICAL |
| 高延迟 | >5s | WARNING |
| 工具错误率高 | >30% | WARNING |
| 验证失败过多 | >10 次 | WARNING |
| 重试过多 | >5 次 | WARNING |

### 业务系统集成

CRM 集成默认使用模拟数据，生产环境设置 `CRM_USE_MOCK=false` 并对接真实 API：

```bash
CRM_USE_MOCK=false
WECOM_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
SMTP_HOST=smtp.qq.com
SMTP_PORT=465
SMTP_USERNAME=your@qq.com
SMTP_PASSWORD=授权码
EMAIL_FROM_ADDR=your@qq.com
```

> **免责声明**：CRM 模块在 `CRM_USE_MOCK=true`（默认）时使用模拟数据，包含的姓名（如张伟、李娜）、公司名（如阿里巴巴、腾讯）、邮箱（@example.com）、电话（13800138000 测试号段）均为示例数据，与任何真实个人或组织无关。生产环境请配置 `CRM_USE_MOCK=false` 对接您的真实 CRM API。

## 项目结构

```
cayz-agent/
├── cayz_agent/                 # 主包
│   ├── __init__.py            # 包入口，导出公共 API
│   ├── __main__.py            # CLI 入口
│   ├── config.py              # 配置管理（pydantic-settings + 启动校验）
│   ├── llm.py                 # 多模型工厂（5 个 provider）
│   ├── graph.py               # 单 Agent LangGraph 图（含 LLM 降级）
│   ├── multi_agent.py         # 多 Agent 协作架构
│   ├── tools.py               # Agent 工具集（25 个工具，三级权限分级）
│   ├── rag.py                 # RAG 知识库管理器（原子更新 + 缓存）
│   ├── api.py                 # FastAPI REST API（含启动诊断报告）
│   ├── middleware.py          # API Key 鉴权 + 限流 + X-Request-ID 中间件
│   ├── app_state.py           # 应用状态管理（就绪标志 + 清理钩子 LIFO）
│   ├── request_context.py     # 请求上下文（contextvars request_id 注入）
│   ├── monitor.py             # Prometheus 指标收集（含 RAG/会话时长）
│   ├── alerts.py              # 阈值告警管理
│   ├── session.py             # 会话管理（列表/查询/删除 + 归属校验）
│   ├── cache.py               # 统一缓存层（per-key 锁防雪崩）
│   ├── exceptions.py          # 自定义异常体系
│   ├── validators.py          # 输入验证
│   ├── sanitizers.py          # 输出脱敏
│   ├── retry.py               # 重试与日志装饰器（异常分类 retryable）
│   └── integrations/          # 业务系统集成
│       ├── __init__.py
│       ├── crm.py             # CRM 客户/订单查询
│       ├── notify.py          # 企业微信 Webhook
│       └── email_sender.py    # SMTP 邮件（连接生命周期管理）
├── scripts/                    # 运维脚本
│   └── backup_sqlite.py       # SQLite 在线备份（WAL checkpoint + 过期清理）
├── tests/                      # 测试（772+ 用例，覆盖率 93%）
│   ├── test_llm.py            # 多模型工厂测试
│   ├── test_rag.py            # RAG 知识库测试
│   ├── test_multi_agent.py    # 多 Agent 架构测试
│   ├── test_graph.py          # 单 Agent 图测试
│   ├── test_tools.py          # 工具测试（9 个原始工具）
│   ├── test_api.py            # API 测试
│   ├── test_middleware.py     # 鉴权 + 限流测试
│   ├── test_monitor.py        # 监控指标测试
│   ├── test_alerts.py         # 告警测试
│   ├── test_session.py        # 会话管理测试
│   ├── test_exceptions.py     # 异常体系测试
│   ├── test_logging.py        # 结构化日志测试
│   ├── test_main.py           # CLI 入口测试
│   ├── test_integration_e2e.py # 端到端集成测试
│   ├── test_integrations_*.py # 业务集成测试（CRM/通知/邮件）
│   ├── test_validators.py     # 验证器测试
│   ├── test_sanitizers.py     # 脱敏测试
│   ├── test_retry.py          # 重试测试
│   ├── test_config.py         # 配置校验测试
│   ├── test_cache.py          # 缓存层测试
│   ├── test_backup_sqlite.py  # SQLite 备份脚本测试
│   ├── test_p3_security.py    # P3 安全测试
│   ├── test_p3_enterprise.py  # P3 企业级部署测试（探针/可观测性/启动报告）
│   ├── test_p3_tools.py       # P3 工具第一梯队测试（calculate/fetch_url/file/repl）
│   ├── test_p3_tools_tier2.py # P3 工具第二梯队测试（PDF/Excel/CSV/二维码/天气/汇率/翻译）
│   └── test_p3_tools_tier3.py # P3 工具第三梯队测试（hash/diff/regex/unit_convert）
├── web_app.py                  # Streamlit Web 界面
├── nginx/                      # Nginx 反向代理配置
│   ├── nginx.conf
│   └── certs/                 # TLS 证书目录（.gitignore 排除）
├── Dockerfile                  # Docker 镜像（多阶段构建 + 非 root）
├── docker-compose.yml          # 多容器编排（含独立数据卷）
├── .github/workflows/ci.yml    # GitHub Actions CI（test/lint/security/docker）
├── pyproject.toml              # 项目元数据与依赖
├── requirements.lock           # 依赖精确版本锁定（145 个包）
├── .env.example                # 环境变量模板
└── README.md                   # 项目文档
```

## 测试

```bash
# 运行全部测试
pytest tests/ -q

# 运行特定模块测试
pytest tests/test_middleware.py -v

# 查看覆盖率
pytest tests/ --cov=cayz_agent
```

当前测试覆盖（共 772+ 测试用例）：

#### 原始模块测试
- `test_llm.py`：12 个测试（5 个 provider + 边界条件）
- `test_rag.py`：30 个测试（文档加载/切片/检索/异常/CachedEmbeddings/单例线程安全/clear/update_document）
- `test_multi_agent.py`：20 个测试（路由/子 Agent/图构建）
- `test_graph.py`：11 个测试
- `test_tools.py`：40 个测试（9 个原始工具全覆盖 + 脱敏）
- `test_api.py`：30+ 个测试（端点/全局异常处理/CORS 顺序/batch-upload 回滚/流式脱敏）
- `test_middleware.py`：20+ 个测试（鉴权 + 限流 + sweep 内存泄漏修复 + X-Request-ID）
- `test_monitor.py`：26+ 个测试（Counter/Histogram/Gauge/导出/RAG 指标）
- `test_alerts.py`：18 个测试（规则/抑制/回调/后台 watcher 线程）
- `test_session.py`：18 个测试（列表/查询/删除/清理/归属校验）
- `test_exceptions.py`：22 个测试（异常继承/属性/可重试标记）
- `test_logging.py`：12 个测试（JSON 格式/extra 字段/异常信息/级别）
- `test_main.py`：7 个测试（CLI 入口：exit/quit/正常输入/异常脱敏/横幅/thread_id）
- `test_integration_e2e.py`：10+ 个测试（端到端：验证/对话/脱敏/工具/会话隔离/LLM 降级）
- `test_integrations_*.py`：30 个测试（CRM/企业微信/邮件集成）
- `test_validators.py`：10 个测试
- `test_sanitizers.py`：17 个测试
- `test_retry.py`：8 个测试
- `test_config.py`：配置校验测试（Pydantic validator + provider-key 一致性）
- `test_cache.py`：缓存层测试（per-key 锁 + 雪崩防护）

#### P2 鲁棒性测试
- `test_p3_security.py`：P3 安全加固测试

#### P3 企业级部署测试
- `test_p3_enterprise.py`：23 个测试（就绪探针/X-Request-ID/启动报告/优雅停机/RAG 指标）
- `test_backup_sqlite.py`：6 个测试（SQLite 在线备份/WAL checkpoint/过期清理）

#### P3 工具扩展测试（3 个梯队共 140 个测试）
- `test_p3_tools.py`：45 个测试（第一梯队：calculate/fetch_url/read_file/write_file/python_repl）
- `test_p3_tools_tier2.py`：38 个测试（第二梯队：parse_pdf/parse_excel/parse_csv/generate_qrcode/get_weather/get_exchange_rate/translate）
- `test_p3_tools_tier3.py`：57 个测试（第三梯队：hash_encode/text_diff/regex_test/unit_convert）

## Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 服务端口：
# - API: http://localhost:8000
# - Web: http://localhost:8501
```

## Agent 工具列表

共 25 个工具，按风险等级三级权限分级，防止 readonly 用户通过 LLM 工具调用绕过 HTTP 层权限。

### readonly 工具（20 个，纯查询/计算，无副作用）

| 工具名 | 功能 | 触发场景 |
|--------|------|---------|
| `get_current_time` | 获取当前时间 | 用户询问时间/日期 |
| `web_search` | 联网搜索（Tavily） | 用户询问实时信息 |
| `knowledge_search` | 知识库检索（RAG） | 用户询问私有知识 |
| `crm_query_customer` | 查询客户信息 | 用户询问客户详情 |
| `crm_search_customers` | 搜索客户 | 按姓名/公司搜索客户 |
| `crm_query_order` | 查询订单详情 | 用户询问订单状态 |
| `calculate` | 安全数学表达式求值（AST 白名单） | 精确计算（避免 LLM 心算出错） |
| `fetch_url` | 网页内容抓取（httpx + readability） | 配合 web_search 读取完整网页 |
| `read_file` | 读取 workspace 内文本文件 | 读取用户上传的文件内容 |
| `python_repl` | 受控 Python 执行（沙箱 + 超时） | 数据分析/字符串处理/复杂计算 |
| `parse_pdf` | PDF 文本提取（pypdf） | 解析 PDF 报告/合同/论文 |
| `parse_excel` | Excel 解析为 Markdown 表格（openpyxl） | 解析 Excel 数据表 |
| `parse_csv` | CSV/TSV 解析为 Markdown 表格 | 解析 CSV 数据导出文件 |
| `get_weather` | 天气查询（和风天气 API） | 天气类问答 |
| `get_exchange_rate` | 汇率查询（fixer.io API） | 汇率/跨境支付场景 |
| `translate` | 翻译（百度翻译 API） | 跨语言沟通 |
| `hash_encode` | 哈希计算 + 编码转换（MD5/SHA/Base64/URL/Hex） | 数据指纹/签名/URL 处理 |
| `text_diff` | 文本差异对比（unified diff） | 文档版本对比/配置变更检查 |
| `regex_test` | 正则表达式测试与匹配提取 | 正则调试/数据提取验证 |
| `unit_convert` | 单位换算（长度/重量/温度/时间/数据量） | 跨境业务/技术文档翻译 |

### write 工具（3 个，有写副作用）

| 工具名 | 功能 | 触发场景 |
|--------|------|---------|
| `knowledge_upload` | 上传知识到知识库 | 用户希望记住信息 |
| `write_file` | 写入 workspace 内文本文件 | LLM 生成内容保存为文件 |
| `generate_qrcode` | 生成二维码图片（离线 qrcode 库） | 生成 URL/文本/名片二维码 |

### admin 工具（2 个，外发副作用，仅管理员可用）

| 工具名 | 功能 | 触发场景 |
|--------|------|---------|
| `send_wecom_notification` | 企业微信通知 | 用户要求发送群通知 |
| `send_email` | 发送邮件 | 用户要求发邮件 |

### 工具安全机制

- **路径白名单**：`read_file` / `write_file` / `parse_*` 限制在 `TOOLS_WORKSPACE_DIR` 目录内，`resolve()` 解析符号链接后校验，防 `../` 越权
- **python_repl 沙箱**：危险关键字黑名单（import/open/exec/eval/subprocess）+ 受限 `__builtins__` + 超时 10s + 输出截断
- **calculate 安全**：AST 白名单节点求值，显式拒绝 Call/Attribute/Subscript，杜绝 `eval` 注入
- **fetch_url 限制**：URL 长度/响应体大小/超时/重定向次数上限
- **外部 API 工具**：未配置 API Key 时返回明确提示，不会崩溃

## 安全特性

### 输入与输出安全
- **输入验证**：长度限制（2000 字符）+ 提示词注入检测（中英文双语模式）
- **输出脱敏**：自动屏蔽 API Key（sk- / tvly- / Bearer / AWS / Aliyun / JWT / 私钥 / 连接串 / 手机号 / 身份证 / 邮箱 / 腾讯云华为云 Key）
- **流式实时脱敏**：SSE 每个 chunk yield 前即时 `sanitize_text`，避免脱敏滞后
- **危险内容检测**：识别 `rm -rf /` 等危险命令
- **日志脱敏**：root logger 安装 `SanitizingLogFilter`，所有日志输出自动脱敏

### 鉴权与访问控制
- **API Key 鉴权**：支持 `X-API-Key` 或 `Authorization: Bearer`，公开端点（/health）免鉴权
- **三级权限分级**：admin / write / readonly，工具集按 scope 分配（见工具列表）
- **会话归属校验**：非管理员仅能访问/删除自己的会话（IDOR 修复）
- **鉴权防裸奔**：`auth_required=True` 时 API_KEY 未配置则非公开端点全部返回 503
- **时序攻击防护**：API Key 比较使用 `hmac.compare_digest()`
- **会话 ID 安全**：`secrets.token_urlsafe(32)` 生成 + 正则校验 `^[A-Za-z0-9_]{8,128}$`

### 限流与防护
- **请求限流**：滑动窗口算法，按 client_id（API Key 或 IP）限流，默认 60 次/分钟，定期 sweep 空闲 client 防内存泄漏
- **写操作独立限流**：`rate_limit_write_per_minute` 施加更严格限制
- **请求体大小限制**：默认 10MB，防 DoS
- **Uvicorn 硬化**：`timeout_keep_alive=5s` 防 Slowloris + `limit_concurrency=100` 防连接耗尽

### 网络与部署安全
- **CORS 控制**：可配置允许的来源，中间件顺序确保鉴权失败响应也带 CORS 头；通配符时强制关闭 credentials
- **HTTPS 强制**：生产环境 HTTP 请求 301 重定向到 HTTPS（/health 豁免）
- **安全响应头**：X-Content-Type-Options / X-Frame-Options / Referrer-Policy / CSP / HSTS（生产）
- **非 root 容器**：Docker 镜像以 uid 1000 的 appuser 运行，`--shell /bin/false` 禁登录
- **API 文档默认关闭**：`docs_enabled=False`，生产环境关闭 /docs /redoc /openapi.json

### 可观测性与运维
- **X-Request-ID 追踪**：中间件生成/透传 request_id，贯穿日志与响应头，便于全链路追踪
- **三层健康探针**：`/health`（存活，公开）+ `/health/ready`（就绪，含依赖检查）+ `/health/deep`（深度，含指标）
- **优雅停机**：SIGTERM 信号处理 + 在途请求等待 + 清理钩子 LIFO 执行（超时 30s）
- **启动诊断报告**：启动时脱敏打印配置摘要 + 依赖预检状态（不阻塞启动）
- **SQLite 在线备份**：WAL checkpoint(TRUNCATE) + Online Backup API + 过期清理脚本

### 工具安全
- **文件路径白名单**：`read_file` / `write_file` / `parse_*` 限制在 `TOOLS_WORKSPACE_DIR` 内，`resolve()` 防符号链接绕过
- **python_repl 沙箱**：危险关键字黑名单 + 受限 `__builtins__` + 超时 10s + 输出截断
- **calculate 安全**：AST 白名单节点求值，杜绝 `eval` 注入
- **知识库敏感检测**：上传文档前扫描敏感信息，支持 off/warn/block 三种模式
- **SMTP 白名单**：收件人域名白名单校验，防止邮件外发到未授权域名
- **Webhook 审核**：企业微信消息发送前自动脱敏

### 错误处理与降级
- **LLM 降级**：LLM 调用失败时返回降级 AIMessage，防止级联失败
- **RAG 原子更新**："先加后删"模式 + 显式 chunk ID，防并发更新数据丢失
- **缓存雪崩防护**：per-key 锁 + double-checked locking，防并发计算碰撞
- **全局异常处理**：`RequestValidationError` → 422，未捕获异常 → 500 + 脱敏（生产模式隐藏内部细节）
- **重试机制**：tenacity 指数退避 + 异常分类（永久性错误跳过重试）

### 供应链安全
- **CI 安全扫描**：pip-audit 依赖 CVE 扫描 + trivy 镜像漏洞扫描（HIGH/CRITICAL 阻断）
- **基础镜像固定**：`python:3.13.7-slim` 具体 patch 版本（builder + runtime 双阶段）
- **依赖锁文件**：`requirements.lock` 精确锁定 145 个包版本，`~=` 兼容约束锁定 minor 版本
- **pre-commit gitleaks**：提交前自动扫描密钥泄露

## 贡献指南

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交前运行 pre-commit 钩子（含 gitleaks 密钥扫描）：`pre-commit install`
4. 确保测试通过：`pytest tests/ -q`
5. 确保覆盖率不低于 80%：`pytest tests/ --cov=cayz_agent`
6. 提交 PR，描述变更内容与动机

## 安全披露

如发现安全漏洞，请**不要**通过 GitHub Issue 公开披露。请发送邮件至 3317866382@qq.com（替换为您的邮箱），我会在 48 小时内响应。

## License

本项目采用 [MIT License](LICENSE) 开源协议。

Copyright (c) 2026 CAYZ Technology
