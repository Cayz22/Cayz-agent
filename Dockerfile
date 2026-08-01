# ===== 阶段 1: 构建阶段（安装依赖到独立目录）=====
# P3 安全修复：固定基础镜像到具体 patch 版本（非浮动 tag），确保可复现构建
# 生产环境建议进一步用 digest pin：
#   FROM python:3.13.7-slim@sha256:<digest>
# 可通过 docker pull python:3.13.7-slim 后 docker inspect --format='{{.RepoDigests}}' 获取 digest
FROM python:3.13.7-slim AS builder

WORKDIR /build

# 系统依赖（仅构建时需要 gcc 编译 C 扩展）
# P3：--no-install-recommends 避免安装推荐包；安装后清理 apt 缓存减小镜像体积
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖声明文件
# P3：使用锁文件完全固定依赖版本，确保 Docker 构建可复现（供应链攻击防护）
# 哈希校验升级：用 pip-compile --generate-hashes 重新生成 requirements.lock 后，
#   在下方 pip install 命令中添加 --require-hashes 即可启用逐包哈希验证。
COPY pyproject.toml requirements.lock ./

# 安装依赖到 /build/venv（不污染系统 Python）
# P3：--no-cache-dir 不缓存 pip 包减小镜像体积；锁文件确保版本完全固定
# 安全修复：requirements.lock 已固定 setuptools==83.0.0 和 msgpack==1.2.1（安全版本），
#   但 venv 创建时自带 setuptools 70.3.0（CVE-2025-47273），msgpack 作为 CacheControl
#   传递依赖可能被解析为旧版 1.1.2（GHSA-6v7p-g79w-8964）。
RUN python -m venv /build/venv \
    && /build/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /build/venv/bin/pip install --no-cache-dir -r requirements.lock

# 复制项目代码并安装包
COPY cayz_agent/ ./cayz_agent/
COPY web_app.py ./
RUN /build/venv/bin/pip install --no-cache-dir --no-deps -e . \
    # 强制卸载并重装漏洞版本，放在所有安装之后确保最终状态正确
    # --force-reinstall 不清理旧 dist-info，Trivy 仍会扫到旧目录名
    && /build/venv/bin/pip uninstall -y setuptools msgpack \
    && /build/venv/bin/pip install --no-cache-dir --no-deps "setuptools==83.0.0" "msgpack==1.2.1"


# ===== 阶段 2: 运行阶段（精简镜像，不含 gcc/构建工具）=====
# P3：运行阶段同样固定到具体 patch 版本
FROM python:3.13.7-slim AS runtime

WORKDIR /app

# 安全加固：更新系统包以修复已知漏洞（Trivy 镜像扫描前置条件）
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制已安装好依赖的 venv
COPY --from=builder /build/venv /app/venv
# 从构建阶段复制应用代码
COPY --from=builder /build/cayz_agent /app/cayz_agent
COPY --from=builder /build/web_app.py /app/
COPY pyproject.toml /app/

# 安全修复验证：扫描镜像中的 setuptools/msgpack 版本
# 应用实际依赖（venv）：setuptools==83.0.0, msgpack==1.2.1（安全版本，见 requirements.lock）
# Trivy 报告的旧版本来自 pip/_vendor 内置副本（pip 上游 vendor 策略，不可控制），
# 已通过 .trivyignore 声明不可利用（vendor 副本仅供 pip 内部使用，应用不 import）
RUN echo "=== 诊断：setuptools/msgpack 安装情况 ===" \
    && find /usr/local/lib/python3.13 /app/venv \
       \( -name "setuptools-*.dist-info" -o -name "msgpack-*.dist-info" \) 2>/dev/null \
       | sort

# 将 venv 加入 PATH
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 数据持久化目录
# P0 修复：预先创建所有挂载子目录并 chown，避免命名卷首次挂载时 Docker 以 root 创建目录
# 导致 appuser(uid 1000) 无法写入（SQLite checkpoint / ChromaDB / 备份全部失败）
RUN mkdir -p /data/checkpoints /data/chroma_db /data/backups

# 创建非 root 用户并以该用户运行（避免容器逃逸时直接获得 root 权限）
# P3：--create-home 创建家目录（部分库需要 HOME 环境变量）；--shell /bin/false 禁止登录
RUN useradd --create-home --shell /bin/false --uid 1000 appuser \
    && chown -R appuser:appuser /app /data
USER appuser

VOLUME ["/data"]

# 暴露 API 端口
EXPOSE 8000

# 默认启动 API 服务（可通过 CMD 覆盖为 streamlit）
CMD ["python", "-m", "cayz_agent.api"]
