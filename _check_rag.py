from cayz_agent.rag import get_rag_manager

manager = get_rag_manager()
results = manager.search("我的名字是什么")
print(f"Results: {len(results)}")
for i, doc in enumerate(results):
    print(f"  [{i}] score={doc.metadata.get('score', '?')}, content={doc.page_content[:100]}")