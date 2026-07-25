"""
rag 模块单元测试：验证知识库管理器

完全 mock 掉 ChromaDB 和 Embeddings，避免真实文件 I/O 和 API 调用
"""
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from cayz_agent.rag import RAGManager
from cayz_agent.config import Settings


@pytest.fixture
def mock_settings():
    """模拟配置"""
    return Settings(
        chroma_persist_dir="./test_chroma_tmp",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        chunk_size=100,
        chunk_overlap=10,
        rag_top_k=3,
        openai_api_key="sk-test",
        openai_api_base="https://api.openai.com/v1",
    )


@pytest.fixture
def mock_embeddings():
    """模拟 Embeddings，返回固定向量"""
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1] * 384
    embeddings.embed_documents.return_value = [[0.1] * 384]
    return embeddings


@pytest.fixture
def mock_vectorstore():
    """模拟 Chroma 向量库"""
    return MagicMock()


class TestRAGManager:
    """测试 RAGManager"""

    def _create_manager(self, mock_settings, mock_embeddings, mock_vectorstore):
        """创建完全 mock 的 RAGManager，不触碰真实文件系统"""
        with patch("cayz_agent.rag.get_settings", return_value=mock_settings), \
             patch("cayz_agent.rag.RAGManager._create_embeddings", return_value=mock_embeddings), \
             patch("cayz_agent.rag.RAGManager._create_vectorstore", return_value=mock_vectorstore):
            return RAGManager()

    def test_init_creates_vectorstore(self, mock_settings, mock_embeddings, mock_vectorstore):
        """初始化时应创建向量库"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        assert manager._embeddings is not None
        assert manager._vectorstore is not None

    def test_add_documents_returns_chunk_count(self, mock_settings, mock_embeddings, mock_vectorstore):
        """add_documents 应返回切片数量"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        long_text = "这是第一段内容。" * 50 + "这是第二段内容。" * 50
        count = manager.add_documents(long_text, source="test")
        assert count > 0

    def test_add_documents_empty_text(self, mock_settings, mock_embeddings, mock_vectorstore):
        """空文本应返回 0"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        count = manager.add_documents("", source="test")
        assert count == 0

    def test_add_documents_whitespace_only(self, mock_settings, mock_embeddings, mock_vectorstore):
        """纯空白文本应返回 0"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        count = manager.add_documents("   \n\n  \t  ", source="test")
        assert count == 0

    def test_search_empty_query(self, mock_settings, mock_embeddings, mock_vectorstore):
        """空查询应返回空列表"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        results = manager.search("")
        assert results == []

    def test_search_whitespace_query(self, mock_settings, mock_embeddings, mock_vectorstore):
        """纯空白查询应返回空列表"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        results = manager.search("   ")
        assert results == []

    def test_search_returns_results(self, mock_settings, mock_embeddings, mock_vectorstore):
        """search 应返回检索结果"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_docs = [Document(page_content="测试内容", metadata={"source": "test"})]
        mock_vectorstore.similarity_search.return_value = mock_docs

        results = manager.search("查询")
        assert len(results) == 1
        assert results[0].page_content == "测试内容"

    def test_search_with_scores(self, mock_settings, mock_embeddings, mock_vectorstore):
        """search_with_scores 应返回带分数的结果"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_results = [(Document(page_content="测试", metadata={}), 0.85)]
        mock_vectorstore.similarity_search_with_score.return_value = mock_results

        results = manager.search_with_scores("查询")
        assert len(results) == 1
        assert isinstance(results[0], tuple)
        assert results[0][1] == 0.85

    def test_search_with_scores_empty_query(self, mock_settings, mock_embeddings, mock_vectorstore):
        """空查询的 search_with_scores 应返回空列表"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        results = manager.search_with_scores("")
        assert results == []

    def test_add_file_txt(self, mock_settings, mock_embeddings, mock_vectorstore, tmp_path):
        """add_file 应能加载 .txt 文件"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        test_file = tmp_path / "test.txt"
        test_file.write_text("这是一段测试文本内容。" * 20, encoding="utf-8")

        count = manager.add_file(str(test_file))
        assert count > 0

    def test_add_file_md(self, mock_settings, mock_embeddings, mock_vectorstore, tmp_path):
        """add_file 应能加载 .md 文件"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        test_file = tmp_path / "test.md"
        test_file.write_text("# 标题\n\n这是 Markdown 内容。" * 20, encoding="utf-8")

        count = manager.add_file(str(test_file))
        assert count > 0

    def test_add_file_not_found(self, mock_settings, mock_embeddings, mock_vectorstore):
        """add_file 文件不存在时应抛出 FileNotFoundError"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        with pytest.raises(FileNotFoundError):
            manager.add_file("/nonexistent/file.txt")

    def test_add_file_unsupported_type(self, mock_settings, mock_embeddings, mock_vectorstore, tmp_path):
        """add_file 不支持的文件类型应抛出 ValueError"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content", encoding="utf-8")

        with pytest.raises(ValueError):
            manager.add_file(str(test_file))

    def test_count(self, mock_settings, mock_embeddings, mock_vectorstore):
        """count 应返回文档数量"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_vectorstore._collection.count.return_value = 42
        assert manager.count() == 42

    def test_count_exception_returns_zero(self, mock_settings, mock_embeddings, mock_vectorstore):
        """count 异常时应返回 0"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_vectorstore._collection.count.side_effect = Exception("DB error")
        assert manager.count() == 0


class TestRAGDeleteBySource:
    """测试 delete_by_source 方法"""

    def _create_manager(self, mock_settings, mock_embeddings, mock_vectorstore):
        with patch("cayz_agent.rag.get_settings", return_value=mock_settings), \
             patch("cayz_agent.rag.RAGManager._create_embeddings", return_value=mock_embeddings), \
             patch("cayz_agent.rag.RAGManager._create_vectorstore", return_value=mock_vectorstore):
            return RAGManager()

    def test_delete_existing_docs(self, mock_settings, mock_embeddings, mock_vectorstore):
        """删除存在的文档应返回删除数量"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_collection = mock_vectorstore._collection
        mock_collection.get.return_value = {"ids": ["id-1", "id-2", "id-3"]}

        count = manager.delete_by_source("manual")
        assert count == 3
        mock_collection.delete.assert_called_once_with(ids=["id-1", "id-2", "id-3"])

    def test_delete_nonexistent_returns_zero(self, mock_settings, mock_embeddings, mock_vectorstore):
        """删除不存在的文档返回 0"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_collection = mock_vectorstore._collection
        mock_collection.get.return_value = {"ids": []}

        count = manager.delete_by_source("nonexistent")
        assert count == 0
        mock_collection.delete.assert_not_called()

    def test_delete_empty_source_returns_zero(self, mock_settings, mock_embeddings, mock_vectorstore):
        """空 source 返回 0"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        assert manager.delete_by_source("") == 0

    def test_delete_raises_rag_error_on_exception(self, mock_settings, mock_embeddings, mock_vectorstore):
        """底层异常应包装为 RAGError"""
        from cayz_agent.exceptions import RAGError

        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_collection = mock_vectorstore._collection
        mock_collection.get.side_effect = Exception("DB error")

        with pytest.raises(RAGError):
            manager.delete_by_source("any")


class TestRAGListSources:
    """测试 list_sources 方法"""

    def _create_manager(self, mock_settings, mock_embeddings, mock_vectorstore):
        with patch("cayz_agent.rag.get_settings", return_value=mock_settings), \
             patch("cayz_agent.rag.RAGManager._create_embeddings", return_value=mock_embeddings), \
             patch("cayz_agent.rag.RAGManager._create_vectorstore", return_value=mock_vectorstore):
            return RAGManager()

    def test_returns_unique_sources(self, mock_settings, mock_embeddings, mock_vectorstore):
        """应返回去重后的 source 列表"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_collection = mock_vectorstore._collection
        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "doc1"},
                {"source": "doc2"},
                {"source": "doc1"},  # 重复
                {"source": "doc3"},
            ]
        }

        sources = manager.list_sources()
        assert sources == ["doc1", "doc2", "doc3"]

    def test_handles_missing_source_field(self, mock_settings, mock_embeddings, mock_vectorstore):
        """缺少 source 字段的 metadata 应被跳过"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_collection = mock_vectorstore._collection
        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "doc1"},
                {"other": "value"},  # 无 source
                None,  # 空 metadata
            ]
        }

        sources = manager.list_sources()
        assert sources == ["doc1"]

    def test_exception_returns_empty_list(self, mock_settings, mock_embeddings, mock_vectorstore):
        """异常时返回空列表"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_collection = mock_vectorstore._collection
        mock_collection.get.side_effect = Exception("DB error")

        assert manager.list_sources() == []


class TestRAGBatchImport:
    """测试 add_batch 方法"""

    def _create_manager(self, mock_settings, mock_embeddings, mock_vectorstore):
        with patch("cayz_agent.rag.get_settings", return_value=mock_settings), \
             patch("cayz_agent.rag.RAGManager._create_embeddings", return_value=mock_embeddings), \
             patch("cayz_agent.rag.RAGManager._create_vectorstore", return_value=mock_vectorstore):
            return RAGManager()

    def test_batch_import_multiple_docs(self, mock_settings, mock_embeddings, mock_vectorstore):
        """批量导入多个文档应返回总片段数"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        long_text = "这是文档内容。" * 30

        items = [
            {"text": long_text, "source": "doc1"},
            {"text": long_text, "source": "doc2"},
        ]
        result = manager.add_batch(items)
        assert result["total"] > 0
        assert result["failed_count"] == 0

    def test_batch_import_empty_list_returns_zero(self, mock_settings, mock_embeddings, mock_vectorstore):
        """空列表返回 total=0"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        result = manager.add_batch([])
        assert result["total"] == 0
        assert result["failed_count"] == 0

    def test_batch_import_skips_empty_text(self, mock_settings, mock_embeddings, mock_vectorstore):
        """空文本应被跳过"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        items = [
            {"text": "", "source": "empty"},
            {"text": "   ", "source": "whitespace"},
        ]
        result = manager.add_batch(items)
        assert result["total"] == 0
        assert result["failed_count"] == 0

    def test_batch_import_continues_on_error(self, mock_settings, mock_embeddings, mock_vectorstore):
        """单个文档失败不应中断整个批量导入"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        long_text = "有效内容。" * 30

        # 第一个文档正常，第二个 mock 失败，第三个正常
        call_count = [0]
        original_add = manager.add_documents

        def side_effect(text, source="batch_import"):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("simulated failure")
            return original_add(text, source=source)

        manager.add_documents = side_effect
        items = [
            {"text": long_text, "source": "doc1"},
            {"text": long_text, "source": "doc2-fail"},
            {"text": long_text, "source": "doc3"},
        ]
        result = manager.add_batch(items)
        # 第一个和第三个成功，第二个失败但不应中断
        assert result["total"] > 0
        assert result["failed_count"] == 1
        assert result["failed"][0]["source"] == "doc2-fail"


class TestRAGUpdateDocument:
    """测试 update_document 方法"""

    def _create_manager(self, mock_settings, mock_embeddings, mock_vectorstore):
        with patch("cayz_agent.rag.get_settings", return_value=mock_settings), \
             patch("cayz_agent.rag.RAGManager._create_embeddings", return_value=mock_embeddings), \
             patch("cayz_agent.rag.RAGManager._create_vectorstore", return_value=mock_vectorstore):
            return RAGManager()

    def test_update_replaces_document(self, mock_settings, mock_embeddings, mock_vectorstore):
        """更新应先删后增"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_collection = mock_vectorstore._collection
        mock_collection.get.return_value = {"ids": ["old-1", "old-2"]}

        long_text = "这是新的文档内容。" * 30
        count = manager.update_document("doc1", long_text)

        # 应先调用 delete（删除旧文档）
        mock_collection.delete.assert_called_once_with(ids=["old-1", "old-2"])
        # 应返回新切片数量
        assert count > 0

    def test_update_empty_source_returns_zero(self, mock_settings, mock_embeddings, mock_vectorstore):
        """空 source 返回 0"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        assert manager.update_document("", "text") == 0

    def test_update_empty_text_returns_zero(self, mock_settings, mock_embeddings, mock_vectorstore):
        """空 text 返回 0"""
        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        assert manager.update_document("src", "") == 0


class TestRAGSingleton:
    """P3 新增：测试 get_rag_manager 单例的线程安全性"""

    def test_get_rag_manager_returns_singleton(self):
        """get_rag_manager 应返回同一实例"""
        import threading
        from cayz_agent.rag import get_rag_manager, reset_rag_manager, RAGManager

        reset_rag_manager()

        with patch("cayz_agent.rag.RAGManager.__init__", return_value=None), \
             patch("cayz_agent.rag.RAGManager._create_embeddings", return_value=MagicMock()), \
             patch("cayz_agent.rag.RAGManager._create_vectorstore", return_value=MagicMock()):
            m1 = get_rag_manager()
            m2 = get_rag_manager()
            assert m1 is m2

        reset_rag_manager()

    def test_get_rag_manager_thread_safe(self):
        """P3 新增：并发调用 get_rag_manager 应只创建一个实例"""
        import threading
        from cayz_agent.rag import get_rag_manager, reset_rag_manager

        reset_rag_manager()

        instances = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            m = get_rag_manager()
            instances.append(m)

        with patch("cayz_agent.rag.RAGManager.__init__", return_value=None), \
             patch("cayz_agent.rag.RAGManager._create_embeddings", return_value=MagicMock()), \
             patch("cayz_agent.rag.RAGManager._create_vectorstore", return_value=MagicMock()):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # 10 个线程拿到的应是同一个实例
        first = instances[0]
        assert all(inst is first for inst in instances)

        reset_rag_manager()

    def test_reset_rag_manager_clears_singleton(self):
        """reset_rag_manager 应清除单例，下次 get_rag_manager 创建新实例"""
        from cayz_agent.rag import get_rag_manager, reset_rag_manager

        reset_rag_manager()

        with patch("cayz_agent.rag.RAGManager.__init__", return_value=None), \
             patch("cayz_agent.rag.RAGManager._create_embeddings", return_value=MagicMock()), \
             patch("cayz_agent.rag.RAGManager._create_vectorstore", return_value=MagicMock()):
            m1 = get_rag_manager()
            reset_rag_manager()
            m2 = get_rag_manager()
            assert m1 is not m2

        reset_rag_manager()


# ============================================================
# P4 新增：CachedEmbeddings 测试
# ============================================================

class TestCachedEmbeddings:
    """P4 新增：测试 CachedEmbeddings 缓存包装器"""

    def test_embed_query_caches_on_miss(self):
        """未命中缓存时应调用底层并写入缓存"""
        from cayz_agent.rag import CachedEmbeddings
        from cayz_agent.cache import reset_cache_singletons

        reset_cache_singletons()

        underlying = MagicMock()
        underlying.embed_query.return_value = [0.1, 0.2, 0.3]

        cached = CachedEmbeddings(underlying)
        result = cached.embed_query("test query")

        assert result == [0.1, 0.2, 0.3]
        underlying.embed_query.assert_called_once_with("test query")

    def test_embed_query_hits_cache(self):
        """命中缓存时不应调用底层"""
        from cayz_agent.rag import CachedEmbeddings
        from cayz_agent.cache import reset_cache_singletons

        reset_cache_singletons()

        underlying = MagicMock()
        underlying.embed_query.return_value = [0.1, 0.2]

        cached = CachedEmbeddings(underlying)
        # 第一次：未命中，调用底层
        r1 = cached.embed_query("same query")
        # 第二次：命中缓存，不调用底层
        r2 = cached.embed_query("same query")

        assert r1 == r2 == [0.1, 0.2]
        assert underlying.embed_query.call_count == 1

    def test_embed_documents_caches_per_text(self):
        """批量向量化应逐条缓存"""
        from cayz_agent.rag import CachedEmbeddings
        from cayz_agent.cache import reset_cache_singletons

        reset_cache_singletons()

        underlying = MagicMock()
        underlying.embed_documents.return_value = [[0.1], [0.2], [0.3]]

        cached = CachedEmbeddings(underlying)
        results = cached.embed_documents(["text1", "text2", "text3"])

        assert results == [[0.1], [0.2], [0.3]]
        underlying.embed_documents.assert_called_once_with(["text1", "text2", "text3"])

    def test_embed_documents_partial_cache_hit(self):
        """部分命中缓存时应只向底层请求未命中的文本"""
        from cayz_agent.rag import CachedEmbeddings
        from cayz_agent.cache import reset_cache_singletons

        reset_cache_singletons()

        underlying = MagicMock()
        underlying.embed_query.return_value = [0.9]  # 预热用

        cached = CachedEmbeddings(underlying)
        # 预热：先单独查 text1，让它进缓存
        cached.embed_query("text1")

        # 重置 mock 计数
        underlying.reset_mock()
        underlying.embed_documents.return_value = [[0.2], [0.3]]

        # 批量查 3 条，text1 已在缓存中
        results = cached.embed_documents(["text1", "text2", "text3"])

        assert len(results) == 3
        # 底层 embed_documents 应只被调用一次，且只传入未命中的 2 条
        underlying.embed_documents.assert_called_once_with(["text2", "text3"])

    def test_embed_documents_all_cached(self):
        """全部命中缓存时不应调用底层"""
        from cayz_agent.rag import CachedEmbeddings
        from cayz_agent.cache import reset_cache_singletons

        reset_cache_singletons()

        underlying = MagicMock()
        underlying.embed_query.return_value = [0.5]

        cached = CachedEmbeddings(underlying)
        # 预热两条
        cached.embed_query("a")
        cached.embed_query("b")

        underlying.reset_mock()
        # 批量查询，两条都在缓存中
        results = cached.embed_documents(["a", "b"])

        assert len(results) == 2
        underlying.embed_documents.assert_not_called()


# ============================================================
# P4 新增：_create_embeddings / _create_vectorstore 分支测试
# ============================================================

class TestRAGCreateEmbeddings:
    """P4 新增：测试 _create_embeddings 的 provider 分支"""

    def test_create_embeddings_ollama_with_langchain_ollama(self):
        """provider=ollama 且已安装 langchain_ollama 时应使用 OllamaEmbeddings"""
        mock_settings = Settings(
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            ollama_base_url="http://localhost:11434",
            chroma_persist_dir="./test_chroma_tmp",
            openai_api_key="sk-test",
            openai_api_base="https://api.openai.com/v1",
        )

        mock_ollama = MagicMock()
        with patch.dict("sys.modules", {"langchain_ollama": MagicMock(OllamaEmbeddings=mock_ollama)}):
            manager = RAGManager.__new__(RAGManager)
            manager._settings = mock_settings
            result = manager._create_embeddings()

        # 应返回 CachedEmbeddings 包装
        from cayz_agent.rag import CachedEmbeddings
        assert isinstance(result, CachedEmbeddings)
        mock_ollama.assert_called_once()

    def test_create_embeddings_ollama_fallback_to_openai(self):
        """provider=ollama 但未安装 langchain_ollama 时应回退到 OpenAI"""
        mock_settings = Settings(
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            ollama_base_url="http://localhost:11434",
            chroma_persist_dir="./test_chroma_tmp",
            openai_api_key="sk-test",
            openai_api_base="https://api.openai.com/v1",
        )

        # 模拟 ImportError
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "langchain_ollama":
                raise ImportError("No module named 'langchain_ollama'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            manager = RAGManager.__new__(RAGManager)
            manager._settings = mock_settings
            result = manager._create_embeddings()

        from cayz_agent.rag import CachedEmbeddings
        assert isinstance(result, CachedEmbeddings)

    def test_create_embeddings_openai_default(self):
        """provider=openai 应直接使用 OpenAIEmbeddings"""
        mock_settings = Settings(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            chroma_persist_dir="./test_chroma_tmp",
            openai_api_key="sk-test",
            openai_api_base="https://api.openai.com/v1",
        )

        manager = RAGManager.__new__(RAGManager)
        manager._settings = mock_settings
        result = manager._create_embeddings()

        from cayz_agent.rag import CachedEmbeddings
        assert isinstance(result, CachedEmbeddings)


class TestRAGCreateVectorstore:
    """P4 新增：测试 _create_vectorstore 的错误处理"""

    def test_create_vectorstore_import_error(self):
        """chromadb 未安装时应抛出 ImportError"""
        mock_settings = Settings(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            chroma_persist_dir="./test_chroma_tmp",
            openai_api_key="sk-test",
            openai_api_base="https://api.openai.com/v1",
        )

        manager = RAGManager.__new__(RAGManager)
        manager._settings = mock_settings

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("chromadb", "langchain_chroma"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError) as exc_info:
                manager._create_vectorstore()

        assert "chromadb" in str(exc_info.value)

    def test_init_raises_rag_connection_error_on_vectorstore_failure(self):
        """_create_vectorstore 失败时应包装为 RAGConnectionError"""
        from cayz_agent.exceptions import RAGConnectionError

        mock_settings = Settings(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            chroma_persist_dir="./test_chroma_tmp",
            openai_api_key="sk-test",
            openai_api_base="https://api.openai.com/v1",
        )

        mock_embeddings = MagicMock()

        with patch("cayz_agent.rag.get_settings", return_value=mock_settings), \
             patch("cayz_agent.rag.RAGManager._create_embeddings", return_value=mock_embeddings), \
             patch("cayz_agent.rag.RAGManager._create_vectorstore", side_effect=RuntimeError("DB init failed")):
            with pytest.raises(RAGConnectionError) as exc_info:
                RAGManager()

        assert "ChromaDB" in str(exc_info.value) or "向量库" in str(exc_info.value)


# ============================================================
# P4 新增：search 缓存命中与 clear 方法测试
# ============================================================

class TestRAGSearchCache:
    """P4 新增：测试 RAG 检索缓存"""

    def test_search_cache_hit(self, mock_settings, mock_embeddings, mock_vectorstore):
        """相同 query 第二次查询应命中缓存，不调用 vectorstore"""
        from cayz_agent.cache import reset_cache_singletons

        reset_cache_singletons()

        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_docs = [Document(page_content="cached", metadata={})]
        mock_vectorstore.similarity_search.return_value = mock_docs

        # 第一次：未命中缓存
        r1 = manager.search("same query")
        assert r1 == mock_docs
        assert mock_vectorstore.similarity_search.call_count == 1

        # 第二次：命中缓存，不调用 vectorstore
        r2 = manager.search("same query")
        assert r2 == mock_docs
        assert mock_vectorstore.similarity_search.call_count == 1  # 仍是 1

    def test_search_with_scores_cache_hit(self, mock_settings, mock_embeddings, mock_vectorstore):
        """search_with_scores 也应缓存"""
        from cayz_agent.cache import reset_cache_singletons

        reset_cache_singletons()

        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        mock_results = [(Document(page_content="x", metadata={}), 0.9)]
        mock_vectorstore.similarity_search_with_score.return_value = mock_results

        r1 = manager.search_with_scores("query")
        r2 = manager.search_with_scores("query")

        assert r1 == r2 == mock_results
        assert mock_vectorstore.similarity_search_with_score.call_count == 1

    def _create_manager(self, mock_settings, mock_embeddings, mock_vectorstore):
        with patch("cayz_agent.rag.get_settings", return_value=mock_settings), \
             patch("cayz_agent.rag.RAGManager._create_embeddings", return_value=mock_embeddings), \
             patch("cayz_agent.rag.RAGManager._create_vectorstore", return_value=mock_vectorstore):
            return RAGManager()


class TestRAGClear:
    """P4 新增：测试 clear 方法"""

    def test_clear_deletes_and_recreates_collection(self, mock_settings, mock_embeddings, mock_vectorstore):
        """clear 应删除 collection 并重新创建"""
        from cayz_agent.cache import reset_cache_singletons

        reset_cache_singletons()

        manager = self._create_manager(mock_settings, mock_embeddings, mock_vectorstore)
        new_vectorstore = MagicMock()
        manager._create_vectorstore = MagicMock(return_value=new_vectorstore)

        manager.clear()

        mock_vectorstore.delete_collection.assert_called_once()
        # 应重新创建 vectorstore
        assert manager._vectorstore is new_vectorstore

    def _create_manager(self, mock_settings, mock_embeddings, mock_vectorstore):
        with patch("cayz_agent.rag.get_settings", return_value=mock_settings), \
             patch("cayz_agent.rag.RAGManager._create_embeddings", return_value=mock_embeddings), \
             patch("cayz_agent.rag.RAGManager._create_vectorstore", return_value=mock_vectorstore):
            return RAGManager()

