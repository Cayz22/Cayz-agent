"""
RAG（检索增强生成）子系统

完整流程：
1. 文档加载：支持 .txt / .md / .pdf
2. 文档切片：按字符长度切片，支持 overlap
3. 向量化：使用 OpenAI Embeddings 或本地 Embeddings
4. 存储：持久化到 ChromaDB
5. 检索：根据 query 向量检索 top_k 最相关文档片段
"""

import hashlib
import logging
import os
import threading
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .cache import get_embedding_cache, get_rag_cache, invalidate_rag_cache
from .config import get_settings
from .exceptions import RAGConnectionError, RAGError, RAGIngestError

logger = logging.getLogger(__name__)


class CachedEmbeddings(Embeddings):
    """带 TTL+LRU 缓存的 Embeddings 包装器。

    包装底层 Embeddings 实例，对 embed_documents / embed_query 的输入文本做哈希缓存。
    适用于：
    - 相同文档重复入库（add_documents 时切片重新向量化）
    - 相同查询重复检索（search 时 query 向量化）

    缓存 key 使用 sha256(text) 避免长文本作为字典 key 的开销。
    """

    def __init__(self, underlying: Embeddings):
        self._underlying = underlying
        self._cache = get_embedding_cache()

    def _key(self, text: str) -> str:
        return "emb:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，逐条查缓存，未命中的批量调用底层。"""
        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(self._key(text))
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            # 批量调用底层 Embeddings（一次 API 请求处理多个文本）
            new_vectors = self._underlying.embed_documents(miss_texts)
            for idx, text, vec in zip(miss_indices, miss_texts, new_vectors, strict=False):
                results[idx] = vec
                self._cache.set(self._key(text), vec)

        return [r for r in results if r is not None]  # type: ignore[list-item]

    def embed_query(self, text: str) -> list[float]:
        """单条查询向量化，走缓存。"""
        cached = self._cache.get(self._key(text))
        if cached is not None:
            return cached
        vec = self._underlying.embed_query(text)
        self._cache.set(self._key(text), vec)
        return vec


class RAGManager:
    """
    RAG 管理器：封装向量数据库的增删查

    Usage:
        manager = RAGManager()
        manager.add_documents("这是一段知识文本", metadata={"source": "manual"})
        results = manager.search("查询问题", top_k=3)
    """

    def __init__(self):
        self._settings = get_settings()
        self._embeddings = self._create_embeddings()
        try:
            self._vectorstore = self._create_vectorstore()
        except Exception as e:
            raise RAGConnectionError("初始化 ChromaDB 向量库失败", cause=e) from e

    def _create_embeddings(self) -> Embeddings:
        """创建 Embeddings 实例（用 CachedEmbeddings 包装以启用缓存）"""
        provider = self._settings.embedding_provider.lower()

        if provider == "ollama":
            try:
                from langchain_ollama import OllamaEmbeddings

                underlying = OllamaEmbeddings(
                    base_url=self._settings.ollama_base_url,
                    model=self._settings.embedding_model,
                )
                return CachedEmbeddings(underlying)
            except ImportError:
                logger.warning("langchain-ollama 未安装，回退到 OpenAI Embeddings")

        # 默认使用 OpenAI Embeddings（DashScope 等非 OpenAI 提供商需禁用 tokenize）
        from langchain_openai import OpenAIEmbeddings

        underlying = OpenAIEmbeddings(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_api_base,
            model=self._settings.embedding_model,
            # 非 OpenAI 提供商（如 DashScope）不支持 token IDs 输入，需发送原始文本
            check_embedding_ctx_length=False,
            # 使用 float 编码避免 base64 兼容性问题
            encoding_format="float",
        )
        return CachedEmbeddings(underlying)

    def _create_vectorstore(self):
        """创建或加载 ChromaDB 向量库"""
        try:
            from langchain_chroma import Chroma
        except ImportError as e:
            raise ImportError(
                "chromadb 和 langchain-chroma 是 RAG 功能的必需依赖，" "请运行: pip install chromadb langchain-chroma"
            ) from e

        persist_dir = self._settings.chroma_persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        logger.info("初始化 ChromaDB, persist_dir=%s", persist_dir)

        return Chroma(
            collection_name="cayz_agent_knowledge",
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )

    def add_documents(
        self,
        text: str,
        metadata: dict | None = None,
        source: str = "manual",
    ) -> int:
        """
        将文本切片后添加到向量库

        Args:
            text: 原始文本
            metadata: 附加元数据
            source: 文档来源标识

        Returns:
            添加的文档片段数量
        """
        return len(self.add_documents_returning_ids(text, metadata=metadata, source=source))

    def add_documents_returning_ids(
        self,
        text: str,
        metadata: dict | None = None,
        source: str = "manual",
    ) -> list[str]:
        """
        将文本切片后添加到向量库，返回新增 chunk 的 ID 列表

        P1 修复：批量上传中途失败回滚时，需按 ID 删除本次新增片段，
        而非按 source 删除（后者会误删该 source 的历史片段）。

        Args:
            text: 原始文本
            metadata: 附加元数据
            source: 文档来源标识

        Returns:
            新增文档片段的 ID 列表（空列表表示未入库）
        """
        if not text or not text.strip():
            logger.warning("添加文档失败：文本为空")
            return []

        metadata = metadata or {}
        metadata.setdefault("source", source)

        doc = Document(page_content=text, metadata=metadata)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )
        chunks = splitter.split_documents([doc])

        if not chunks:
            logger.warning("文档切片结果为空")
            return []

        try:
            # P2-3 修复：改用显式 ID 传入，避免并发上传时差集推断导致 ID 错误归属。
            # 旧实现通过 collection.get() 前后差集推断新增 ID，并发场景下会把其他线程
            # 新增的 ID 误判为本批次，导致回滚时误删其他线程的片段。
            # 显式生成 uuid 后由 ChromaDB 原子写入，无需差集推断。
            new_ids = [str(uuid.uuid4()) for _ in chunks]
            self._vectorstore.add_documents(chunks, ids=new_ids)
        except Exception as e:
            raise RAGIngestError(f"文档入库失败: {e}", cause=e) from e

        # 知识库变更，主动失效 RAG 检索缓存
        invalidate_rag_cache()

        logger.info("添加文档成功: %d 个片段, source=%s", len(chunks), source)
        return new_ids

    def add_file(self, file_path: str) -> int:
        """
        加载文件并添加到向量库

        支持: .txt, .md, .pdf

        Args:
            file_path: 文件路径

        Returns:
            添加的文档片段数量
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        logger.info("加载文件: %s (类型: %s)", file_path, ext)

        if ext == ".pdf":
            text = self._load_pdf(file_path)
        elif ext in (".txt", ".md"):
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError(f"不支持的文件类型: {ext}（支持 .txt/.md/.pdf）")

        return self.add_documents(text, source=file_path)

    def _load_pdf(self, file_path: str) -> str:
        """加载 PDF 文件文本"""
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except ImportError as e:
                raise ImportError("加载 PDF 需要 pypdf 或 PyPDF2，请运行: pip install pypdf") from e

        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return "\n\n".join(texts)

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """
        检索与 query 最相关的文档片段

        Args:
            query: 查询文本
            top_k: 返回数量，默认使用配置中的 rag_top_k

        Returns:
            相关文档片段列表，按相关度降序
        """
        if not query or not query.strip():
            return []

        k = top_k or self._settings.rag_top_k

        # 短期检索缓存（相同 query + top_k 复用结果，降低 ChromaDB 压力）
        cache = get_rag_cache()
        cache_key = f"search:{k}:{hashlib.sha256(query.encode('utf-8')).hexdigest()}"
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info("RAG 检索缓存命中: query='%s'", query[:50])
            # P3：缓存命中也记录检索指标（result_count 反映召回量）
            from .monitor import record_rag_search

            record_rag_search(success=True, latency=0.0, result_count=len(cached))
            return cached

        # P3：记录检索延迟与结果数（异常时也记录）
        import time as _time

        from .monitor import record_rag_search

        _start = _time.perf_counter()
        try:
            results = self._vectorstore.similarity_search(query, k=k)
            _latency = _time.perf_counter() - _start
            logger.info("RAG 检索: query='%s', 返回 %d 个结果", query[:50], len(results))
            record_rag_search(success=True, latency=_latency, result_count=len(results))
            cache.set(cache_key, results)
            return results
        except Exception:
            _latency = _time.perf_counter() - _start
            record_rag_search(success=False, latency=_latency, result_count=0)
            raise

    def search_with_scores(self, query: str, top_k: int | None = None) -> list[tuple[Document, float]]:
        """检索并返回相关度分数"""
        if not query or not query.strip():
            return []

        k = top_k or self._settings.rag_top_k

        # 短期检索缓存
        cache = get_rag_cache()
        cache_key = f"search_scores:{k}:{hashlib.sha256(query.encode('utf-8')).hexdigest()}"
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info("RAG 检索(带分数)缓存命中: query='%s'", query[:50])
            return cached

        results = self._vectorstore.similarity_search_with_score(query, k=k)
        logger.info("RAG 检索(带分数): query='%s', 返回 %d 个结果", query[:50], len(results))
        cache.set(cache_key, results)
        return results

    def clear(self) -> None:
        """清空向量库中的所有文档

        P2-8 修复：先创建新 collection 再删旧，避免重建失败导致管理器永久损坏。
        旧实现先 delete_collection 再 _create_vectorstore，若第二步失败（磁盘满/
        权限变更），_vectorstore 仍指向已删除的 collection，后续所有操作都会失败，
        且单例缓存的损坏实例无法自动恢复。
        """
        old = self._vectorstore
        try:
            # 先建新的空 collection，确保重建成功
            self._vectorstore = self._create_vectorstore()
        except Exception as e:
            # 重建失败：恢复旧引用，保持管理器可用状态
            logger.exception("重建向量库失败，保留旧实例")
            self._vectorstore = old
            raise RAGError(f"清空向量库失败（重建异常）: {e}", cause=e) from e
        # 新 collection 已就绪，安全删除旧 collection
        try:
            old.delete_collection()
        except Exception:
            # 旧 collection 删除失败不影响新实例使用，仅记录警告
            logger.warning("删除旧 collection 失败（新 collection 已就绪，忽略）")
        # 主动失效 RAG 检索缓存
        invalidate_rag_cache()
        logger.info("向量库已清空")

    def count(self) -> int:
        """返回向量库中的文档片段总数。

        P2 修复：异常向上传播，便于 /health/deep 准确探测 ChromaDB 可用性。
        调用方如需容错请使用 count_safe() 或自行 try/except。
        """
        return self._vectorstore._collection.count()

    def count_safe(self) -> int:
        """容错版 count：异常时返回 0 并记录警告，仅供非关键路径使用。"""
        try:
            return self.count()
        except Exception:
            logger.warning("count_safe 容错返回 0", exc_info=True)
            return 0

    def delete_by_source(self, source: str) -> int:
        """
        按来源标识删除文档

        Args:
            source: 文档来源（metadata.source 字段）

        Returns:
            删除的文档片段数量
        """
        if not source:
            return 0

        try:
            collection = self._vectorstore._collection
            # ChromaDB where 条件按 metadata 过滤
            results = collection.get(where={"source": source})
            ids = results.get("ids", []) if results else []
            if ids:
                collection.delete(ids=ids)
                # 主动失效 RAG 检索缓存
                invalidate_rag_cache()
                logger.info("按 source='%s' 删除 %d 个片段", source, len(ids))
                return len(ids)
            return 0
        except Exception as e:
            logger.exception("按 source 删除文档失败")
            raise RAGError(f"按 source 删除文档失败: {e}", cause=e) from e

    def delete_by_ids(self, ids: list[str]) -> int:
        """
        按 chunk ID 列表删除文档

        P1 修复：批量上传中途失败回滚时，按本次新增 chunk 的 ID 精确删除，
        避免误删该 source 下已有的历史片段。

        Args:
            ids: 要删除的 chunk ID 列表

        Returns:
            实际删除的片段数量
        """
        if not ids:
            return 0
        try:
            collection = self._vectorstore._collection
            # 过滤掉占位 ID（unknown_* 是 add_documents_returning_ids 差集为空时的回退）
            real_ids = [i for i in ids if not i.startswith("unknown_")]
            if not real_ids:
                logger.warning("无可删除的真实 chunk ID（均为占位符），跳过回滚")
                return 0
            collection.delete(ids=real_ids)
            invalidate_rag_cache()
            logger.info("按 ID 删除 %d 个片段", len(real_ids))
            return len(real_ids)
        except Exception as e:
            logger.exception("按 ID 删除文档失败")
            raise RAGError(f"按 ID 删除文档失败: {e}", cause=e) from e

    def list_sources(self) -> list[str]:
        """
        列出知识库中所有文档来源

        Returns:
            去重后的 source 列表
        """
        try:
            collection = self._vectorstore._collection
            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", []) if results else []
            sources = set()
            for meta in metadatas:
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return sorted(sources)
        except Exception:
            logger.exception("列出文档来源失败")
            return []

    def get_by_source(self, source: str) -> list[dict]:
        """
        按来源获取所有文档片段

        Args:
            source: 文档来源（metadata.source 字段）

        Returns:
            [{"id": str, "content": str, "metadata": dict}, ...]
        """
        if not source:
            return []
        try:
            collection = self._vectorstore._collection
            results = collection.get(
                where={"source": source},
                include=["documents", "metadatas"],
            )
            items = []
            ids = results.get("ids", []) or []
            docs = results.get("documents", []) or []
            metas = results.get("metadatas", []) or []
            for i, doc_id in enumerate(ids):
                items.append({
                    "id": doc_id,
                    "content": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                })
            return items
        except Exception:
            logger.exception("按 source 获取文档失败")
            return []

    def add_batch(self, items: list[dict]) -> dict:
        """批量导入文档

        P2-13 修复：返回结构化结果，包含成功总数与失败列表，
        避免静默吞没单项失败导致调用方无法感知部分失败。

        Args:
            items: [{"text": "...", "source": "..."}, ...]

        Returns:
            {
                "total": 成功导入的片段总数,
                "succeeded": 成功的 source 列表,
                "failed": [{"source": "...", "error": "..."}, ...],
                "failed_count": 失败项数,
            }
        """
        if not items:
            return {"total": 0, "succeeded": [], "failed": [], "failed_count": 0}

        total = 0
        succeeded: list[str] = []
        failed: list[dict] = []
        for item in items:
            text = item.get("text", "")
            source = item.get("source", "batch_import")
            if text and text.strip():
                try:
                    count = self.add_documents(text, source=source)
                    total += count
                    succeeded.append(source)
                except Exception as e:
                    logger.warning("批量导入文档失败 (source=%s): %s", source, e)
                    failed.append({"source": source, "error": str(e)})

        logger.info(
            "批量导入完成: %d 个文档, 成功 %d, 失败 %d, 共 %d 个片段",
            len(items),
            len(succeeded),
            len(failed),
            total,
        )
        return {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "failed_count": len(failed),
        }

    def update_document(self, source: str, new_text: str) -> int:
        """更新文档（先添加新文档，确认成功后再删除旧文档）

        P2-2 修复：改为"先加后删"模式，避免旧实现"先删后加"在添加失败时
        导致知识库内容永久丢失且无法回滚。
        旧实现先 delete_by_source 删除旧文档，再 add_documents 添加新文档；
        若 add_documents 失败（ChromaDB 异常/embedding 超时），旧文档已删除
        而新文档未写入，知识库内容永久丢失。

        Args:
            source: 要更新的文档来源
            new_text: 新的文档内容

        Returns:
            新切片数量
        """
        if not source or not new_text or not new_text.strip():
            return 0

        # 先添加新文档，记录新增 chunk ID
        new_ids = self.add_documents_returning_ids(new_text, source=source)
        if not new_ids:
            logger.warning("更新文档失败：新文档入库失败，保留旧文档 (source=%s)", source)
            return 0

        # 新文档入库成功后，删除该 source 下的旧片段（排除刚新增的 ID）
        new_id_set = set(new_ids)
        try:
            collection = self._vectorstore._collection
            results = collection.get(where={"source": source})
            all_ids = results.get("ids", []) if results else []
            # 仅删除旧的（不在 new_ids 中的）
            old_ids = [i for i in all_ids if i not in new_id_set]
            if old_ids:
                collection.delete(ids=old_ids)
                invalidate_rag_cache()
                logger.info(
                    "更新文档: 已添加 %d 新片段, 已删除 %d 旧片段 (source=%s)",
                    len(new_ids),
                    len(old_ids),
                    source,
                )
            else:
                logger.info("更新文档: 已添加 %d 新片段, 无旧片段需删除 (source=%s)", len(new_ids), source)
        except Exception as e:
            # 删除旧片段失败不影响新文档可用性，仅记录警告
            logger.warning("更新文档：删除旧片段失败，新文档已入库 (source=%s): %s", source, e)

        return len(new_ids)


# ===== 单例管理 =====
_rag_manager: RAGManager | None = None
_rag_lock = threading.Lock()


def get_rag_manager() -> RAGManager:
    """获取 RAG 管理器单例（线程安全，双重检查锁定）"""
    global _rag_manager
    if _rag_manager is None:
        with _rag_lock:
            # 再次检查，避免在等锁期间已被其他线程创建
            if _rag_manager is None:
                _rag_manager = RAGManager()
    return _rag_manager


def reset_rag_manager() -> None:
    """重置 RAG 管理器单例（主要用于测试场景）"""
    global _rag_manager
    with _rag_lock:
        _rag_manager = None
