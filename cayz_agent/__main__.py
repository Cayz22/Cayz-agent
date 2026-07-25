"""CLI 入口：交互式对话"""
import logging

from langchain_core.messages import HumanMessage

from . import __version__
from .config import get_settings, setup_logging
from .graph import create_graph
from .sanitizers import sanitize_text, sanitize_exception

logger = logging.getLogger(__name__)


def main():
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)

    logger.info("cayz-agent v%s 启动中...", __version__)
    app = create_graph()
    logger.info("cayz-agent v%s 启动成功！", __version__)
    print(f"🤖 [cayz-agent v{__version__}] 启动成功！")
    print("💬 请输入您的指令（输入 'exit' 退出）:\n")

    config = {"configurable": {"thread_id": "cayz-user-session-001"}}

    while True:
        user_input = input(" 你: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 [cayz-agent] 再见！")
            break

        try:
            logger.info("用户输入: %s", user_input)

            # 1. 调用 Agent 获取结果
            result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            ai_message = result["messages"][-1].content

            # 2. 输出安全审查（脱敏）
            safe_message = sanitize_text(ai_message)

            # 3. 任务执行流提示
            print("\n🚀 [任务执行流] 👤接收指令 ➡️ 🧠意图识别 ➡️ 🔍工具调用 ➡️ 📊输出结果")

            # 4. 打印 Agent 的最终回复
            print(f" Agent: {safe_message}\n")

        except Exception as e:
            logger.error("Agent 执行出错: %s", sanitize_exception(e))
            print(f"❌ Agent 执行出错: {sanitize_exception(e)}\n")


if __name__ == "__main__":
    main()
