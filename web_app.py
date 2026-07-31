"""Streamlit Web UI 入口 - GALASKY 深空主题"""
import os
import uuid

import streamlit as st
import requests
from langchain_core.messages import HumanMessage, AIMessageChunk

from cayz_agent import create_graph, setup_logging, get_settings, __version__
from cayz_agent.sanitizers import sanitize_text, sanitize_exception

# 1. 页面基础配置
st.set_page_config(page_title="cayz-agent", page_icon="", layout="wide")

# 延迟初始化日志与配置
_settings = get_settings()
setup_logging(_settings.log_level, _settings.log_format)

# 2. GALASKY 深空主题 CSS
GALASKY_CSS = """
<style>
/* ========== 全局背景 ========== */
.stApp {
    background: #ffffff !important;
    min-height: 100vh !important;
}

/* 隐藏 Streamlit 默认装饰 */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
header { visibility: hidden !important; }

/* ========== 主容器 ========== */
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

[data-testid="stAppViewBlockContainer"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    box-shadow: none !important;
}

/* ========== 标题区域 ========== */
h1 {
    color: #1a1a1a !important;
    font-size: 1.6rem !important;
    font-weight: 300 !important;
    text-align: center !important;
    letter-spacing: 0.15em !important;
    margin-bottom: 0 !important;
    padding-top: 24px !important;
}

/* 副标题 */
[data-testid="stMarkdownContainer"] > p {
    color: rgba(0, 0, 0, 0.4) !important;
    text-align: center !important;
    font-size: 0.82rem !important;
    font-weight: 300 !important;
    margin-bottom: 0 !important;
}

/* ========== 聊天输入框（固定在底部中央） ========== */
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 28px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    max-width: 640px !important;
    margin: 0 auto !important;
    box-shadow: 0 2px 16px rgba(0, 0, 0, 0.06) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(140, 120, 220, 0.5) !important;
    box-shadow: 0 4px 24px rgba(140, 120, 220, 0.12) !important;
}

[data-testid="stChatInput"] input {
    color: #1a1a1a !important;
    font-size: 0.92rem !important;
}

[data-testid="stChatInput"] input::placeholder {
    color: rgba(0, 0, 0, 0.3) !important;
}

/* 输入框容器固定在底部 */
[data-testid="stChatInputContainer"] {
    position: fixed !important;
    bottom: 28px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    z-index: 1000 !important;
    width: 90% !important;
    max-width: 640px !important;
}

/* 底部容器 */
[data-testid="stBottomBlockContainer"] {
    background-color: transparent !important;
    width: 820px !important;
    border-width: 0px !important;
    border-style: solid !important;
    border-color: transparent !important;
    height: 25px !important;
}

/* 底部外层容器 */
.st-emotion-cache-hzygls {
    background-color: transparent !important;
}

/* ========== 聊天消息区域 ========== */
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child {
    padding-bottom: 100px !important;
}

/* ========== 聊天消息（微信风格） ========== */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 4px 0 !important;
    margin-bottom: 16px !important;
    gap: 10px !important;
    align-items: flex-start !important;
}

/* 头像：圆形 */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    width: 38px !important;
    height: 38px !important;
    min-width: 38px !important;
    border-radius: 50% !important;
    overflow: hidden !important;
    border: none !important;
}

[data-testid="stChatMessageAvatarUser"] > span,
[data-testid="stChatMessageAvatarAssistant"] > span {
    width: 38px !important;
    height: 38px !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: none !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.02em !important;
}

/* 用户头像：淡紫底 + "I" */
[data-testid="stChatMessageAvatarUser"] > span {
    background: rgba(140, 120, 220, 0.1) !important;
    color: #6a5a9e !important;
    border: 1.5px solid rgba(140, 120, 220, 0.2) !important;
}
[data-testid="stChatMessageAvatarUser"] > span::after {
    content: 'I' !important;
}
[data-testid="stChatMessageAvatarUser"] [data-testid="stIconMaterial"] {
    display: none !important;
}

/* 助手头像：淡蓝底 + "AI" */
[data-testid="stChatMessageAvatarAssistant"] > span {
    background: rgba(100, 140, 220, 0.1) !important;
    color: #4a6a9e !important;
    border: 1.5px solid rgba(100, 140, 220, 0.2) !important;
}
[data-testid="stChatMessageAvatarAssistant"] > span::after {
    content: 'AI' !important;
}
[data-testid="stChatMessageAvatarAssistant"] [data-testid="stIconMaterial"] {
    display: none !important;
}

/* 内容区及其子层 */
[data-testid="stChatMessageContent"],
[data-testid="stChatMessageContent"] > div,
[data-testid="stChatMessageContent"] .stElementContainer,
[data-testid="stChatMessageContent"] .stMarkdown {
    background: transparent !important;
    width: auto !important;
    max-width: none !important;
    min-width: 0 !important;
    flex: 0 1 auto !important;
    writing-mode: horizontal-tb !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 气泡本体 */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    padding: 10px 16px !important;
    border-radius: 12px !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    position: relative !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    display: inline-block !important;
    writing-mode: horizontal-tb !important;
    max-width: calc(15em + 32px) !important;
    box-sizing: border-box !important;
    min-width: 2em !important;
}

/* 气泡内段落 */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: inherit !important;
    text-align: left !important;
    font-size: 0.92rem !important;
    margin: 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p + p {
    margin-top: 6px !important;
}

/* ---- 助手消息（左对齐，白底气泡） ---- */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    flex-direction: row !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] {
    background: #ffffff !important;
    color: #2a2a2a !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-top-left-radius: 4px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
}

/* ---- 用户消息（右对齐，淡紫气泡） ---- */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
    background: rgba(140, 120, 220, 0.08) !important;
    color: #2a2a2a !important;
    border: 1px solid rgba(140, 120, 220, 0.12) !important;
    border-top-right-radius: 4px !important;
}

/* ========== Spinner 加载动画 ========== */
[data-testid="stSpinner"] > div {
    border-color: rgba(140, 120, 220, 0.15) !important;
    border-top-color: rgba(140, 120, 220, 0.6) !important;
}

/* ========== 错误提示 ========== */
[data-testid="stAlert"] {
    background: rgba(220, 100, 100, 0.05) !important;
    border: 1px solid rgba(220, 100, 100, 0.15) !important;
    border-radius: 12px !important;
    color: #8a3030 !important;
}

/* ========== 滚动条 ========== */
::-webkit-scrollbar {
    width: 4px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.12);
    border-radius: 2px;
}

/* ========== 侧边栏 ========== */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: transparent !important;
}
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1a1a1a !important;
    font-weight: 400 !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stTextArea label {
    color: rgba(0, 0, 0, 0.5) !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background: #fafafa !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    color: #1a1a1a !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stTextInput input:focus,
[data-testid="stSidebar"] .stTextArea textarea:focus {
    border-color: rgba(140, 120, 220, 0.4) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: rgba(0, 0, 0, 0.6) !important;
    border: 1px solid rgba(0, 0, 0, 0.12) !important;
    border-radius: 8px !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(140, 120, 220, 0.06) !important;
    border-color: rgba(140, 120, 220, 0.25) !important;
}
[data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
}
[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(0, 0, 0, 0.35) !important;
    border-radius: 8px !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stTabs [aria-selected="true"] {
    background: rgba(140, 120, 220, 0.06) !important;
    color: rgba(0, 0, 0, 0.7) !important;
}
[data-testid="stSidebar"] .stExpander {
    background: #fafafa !important;
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stExpander summary {
    color: rgba(0, 0, 0, 0.5) !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] code {
    background: rgba(140, 120, 220, 0.06) !important;
    color: #6a5a9e !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
}
[data-testid="stSidebar"] .stDivider {
    border-color: rgba(0, 0, 0, 0.08) !important;
}
[data-testid="stSidebar"] .stCaption {
    color: rgba(0, 0, 0, 0.35) !important;
}
[data-testid="stSidebar"] .stSuccess,
[data-testid="stSidebar"] .stWarning,
[data-testid="stSidebar"] .stError,
[data-testid="stSidebar"] .stInfo {
    background: rgba(140, 120, 220, 0.04) !important;
    border: 1px solid rgba(140, 120, 220, 0.1) !important;
    border-radius: 8px !important;
}

/* ========== 点阵装饰 ========== */
.dot-grid-tr {
    position: fixed;
    top: 30px;
    right: 40px;
    width: 60px;
    height: 60px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.25;
}
.dot-grid-bl {
    position: fixed;
    bottom: 30px;
    left: 40px;
    width: 60px;
    height: 60px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.2;
}

/* ========== 行星装饰 ========== */
.planet-ring {
    position: fixed;
    top: 80px;
    left: 60px;
    width: 80px;
    height: 80px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.15;
}
.planet-blue {
    position: fixed;
    top: 50%;
    right: 80px;
    width: 24px;
    height: 24px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.2;
}
.planet-orange {
    position: fixed;
    bottom: 120px;
    left: 50%;
    transform: translateX(-50%);
    width: 50px;
    height: 50px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.12;
}
.planet-arc {
    position: fixed;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 300px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.08;
}
</style>
"""

st.markdown(GALASKY_CSS, unsafe_allow_html=True)

# 装饰元素 SVG
# 右上角点阵
st.markdown('''
<svg class="dot-grid-tr" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg">
    <g fill="rgba(0,0,0,0.12)">
        <circle cx="6" cy="6" r="1.5"/><circle cx="18" cy="6" r="1.5"/><circle cx="30" cy="6" r="1.5"/><circle cx="42" cy="6" r="1.5"/><circle cx="54" cy="6" r="1.5"/>
        <circle cx="6" cy="18" r="1.5"/><circle cx="18" cy="18" r="1.5"/><circle cx="30" cy="18" r="1.5"/><circle cx="42" cy="18" r="1.5"/><circle cx="54" cy="18" r="1.5"/>
        <circle cx="6" cy="30" r="1.5"/><circle cx="18" cy="30" r="1.5"/><circle cx="30" cy="30" r="1.5"/><circle cx="42" cy="30" r="1.5"/><circle cx="54" cy="30" r="1.5"/>
        <circle cx="6" cy="42" r="1.5"/><circle cx="18" cy="42" r="1.5"/><circle cx="30" cy="42" r="1.5"/><circle cx="42" cy="42" r="1.5"/><circle cx="54" cy="42" r="1.5"/>
        <circle cx="6" cy="54" r="1.5"/><circle cx="18" cy="54" r="1.5"/><circle cx="30" cy="54" r="1.5"/><circle cx="42" cy="54" r="1.5"/><circle cx="54" cy="54" r="1.5"/>
    </g>
</svg>
''', unsafe_allow_html=True)

# 左下角点阵
st.markdown('''
<svg class="dot-grid-bl" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg">
    <g fill="rgba(0,0,0,0.1)">
        <circle cx="6" cy="6" r="1.5"/><circle cx="18" cy="6" r="1.5"/><circle cx="30" cy="6" r="1.5"/><circle cx="42" cy="6" r="1.5"/><circle cx="54" cy="6" r="1.5"/>
        <circle cx="6" cy="18" r="1.5"/><circle cx="18" cy="18" r="1.5"/><circle cx="30" cy="18" r="1.5"/><circle cx="42" cy="18" r="1.5"/><circle cx="54" cy="18" r="1.5"/>
        <circle cx="6" cy="30" r="1.5"/><circle cx="18" cy="30" r="1.5"/><circle cx="30" cy="30" r="1.5"/><circle cx="42" cy="30" r="1.5"/><circle cx="54" cy="30" r="1.5"/>
        <circle cx="6" cy="42" r="1.5"/><circle cx="18" cy="42" r="1.5"/><circle cx="30" cy="42" r="1.5"/><circle cx="42" cy="42" r="1.5"/><circle cx="54" cy="42" r="1.5"/>
        <circle cx="6" cy="54" r="1.5"/><circle cx="18" cy="54" r="1.5"/><circle cx="30" cy="54" r="1.5"/><circle cx="42" cy="54" r="1.5"/><circle cx="54" cy="54" r="1.5"/>
    </g>
</svg>
''', unsafe_allow_html=True)

# 左上角带环行星
st.markdown('''
<svg class="planet-ring" viewBox="0 0 80 80" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <radialGradient id="pg1" cx="40%" cy="35%">
            <stop offset="0%" stop-color="#a855c8"/>
            <stop offset="100%" stop-color="#6a2090"/>
        </radialGradient>
    </defs>
    <circle cx="40" cy="40" r="22" fill="url(#pg1)"/>
    <ellipse cx="40" cy="40" rx="36" ry="10" fill="none" stroke="rgba(0,0,0,0.12)" stroke-width="1.2" transform="rotate(-20 40 40)"/>
</svg>
''', unsafe_allow_html=True)

# 右侧蓝色小球
st.markdown('''
<svg class="planet-blue" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <radialGradient id="pb1" cx="35%" cy="30%">
            <stop offset="0%" stop-color="#6090d0"/>
            <stop offset="100%" stop-color="#2a4080"/>
        </radialGradient>
    </defs>
    <circle cx="12" cy="12" r="10" fill="url(#pb1)"/>
</svg>
''', unsafe_allow_html=True)

# 底部橙色球
st.markdown('''
<svg class="planet-orange" viewBox="0 0 50 50" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <radialGradient id="po1" cx="40%" cy="30%">
            <stop offset="0%" stop-color="#d08040"/>
            <stop offset="100%" stop-color="#804020"/>
        </radialGradient>
    </defs>
    <circle cx="25" cy="25" r="20" fill="url(#po1)"/>
</svg>
''', unsafe_allow_html=True)

# 底部大弧线
st.markdown('''
<svg class="planet-arc" viewBox="0 0 600 300" xmlns="http://www.w3.org/2000/svg">
    <path d="M0 300 Q300 0 600 300" fill="none" stroke="rgba(0,0,0,0.06)" stroke-width="1"/>
</svg>
''', unsafe_allow_html=True)

# 3. 初始化 Agent 图与 Session State
# P1 安全修复：web_app 直连 graph 默认用 admin scope，含全部危险工具（send_email、
# write_file、python_repl 等），且 Streamlit 无原生鉴权。改为 readonly scope：
# 仅含 get_current_time / web_search / knowledge_search / calculate 等查询类工具，
# 写操作（知识库上传/删除、会话删除）一律走侧边栏的 HTTP API 调用（受鉴权保护）
if "app" not in st.session_state:
    st.session_state.app = create_graph(scope="readonly")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web-{uuid.uuid4()}"

# ============================================================
# 侧边栏：管理面板（会话/知识库/健康检查）
# ============================================================
API_BASE = os.environ.get("WEB_API_BASE", f"http://localhost:{_settings.api_port}")

with st.sidebar:
    st.markdown("### 管理面板")

    # ---- API Key 输入 ----
    api_key = st.text_input(
        "API Key",
        value=st.session_state.get("api_key", ""),
        type="password",
        help="如后端启用了 API Key 鉴权，请在此填写；公开端点（/health /metrics）无需填写",
    )
    st.session_state.api_key = api_key
    _headers = {"X-API-Key": api_key} if api_key else {}

    # ---- 标签页：会话 / 知识库 / 健康 ----
    tab_sessions, tab_kb, tab_health = st.tabs(["会话", "知识库", "健康"])

    # ===== 会话管理 =====
    with tab_sessions:
        if st.button("刷新会话列表", use_container_width=True):
            st.session_state._sessions_loaded = False

        try:
            resp = requests.get(
                f"{API_BASE}/sessions?limit=50",
                headers=_headers,
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                sessions = data.get("sessions", [])
                if not sessions:
                    st.info("暂无会话记录")
                else:
                    st.caption(f"共 {data.get('total', len(sessions))} 个会话")
                    for s in sessions:
                        thread_id = s.get("thread_id", "")
                        last_active = s.get("last_active", "")
                        msg_count = s.get("message_count", 0)
                        with st.expander(
                            f"{thread_id[:16]}...  | 消息 {msg_count} 条"
                        ):
                            st.caption(f"完整 ID: `{thread_id}`")
                            st.caption(f"最后活跃: {last_active}")
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("切换", key=f"switch-{thread_id}"):
                                    st.session_state.thread_id = thread_id
                                    st.session_state.messages = []
                                    st.success("已切换会话（请重新加载页面以加载历史）")
                            with col2:
                                if st.button("删除", key=f"del-{thread_id}"):
                                    del_resp = requests.delete(
                                        f"{API_BASE}/sessions/{thread_id}",
                                        headers=_headers,
                                        timeout=5,
                                    )
                                    if del_resp.status_code == 200 and del_resp.json().get("deleted"):
                                        st.success("已删除")
                                        st.session_state._sessions_loaded = False
                                    else:
                                        st.error("删除失败")
            else:
                st.error(f"获取会话失败: HTTP {resp.status_code}")
        except requests.RequestException as e:
            st.warning(f"无法连接 API（{API_BASE}）：{e}")

    # ===== 知识库管理 =====
    with tab_kb:
        st.markdown("**上传文档**")
        with st.form("kb_upload_form", clear_on_submit=True):
            kb_text = st.text_area(
                "文档内容",
                height=150,
                placeholder="粘贴要上传到知识库的文本内容...",
            )
            kb_source = st.text_input(
                "来源标识",
                value="web_ui_upload",
                help="用于后续按来源删除/更新",
            )
            if st.form_submit_button("上传", use_container_width=True):
                if not kb_text.strip():
                    st.error("内容不能为空")
                else:
                    try:
                        resp = requests.post(
                            f"{API_BASE}/knowledge/upload",
                            json={"text": kb_text, "source": kb_source},
                            headers=_headers,
                            timeout=30,
                        )
                        if resp.status_code == 200:
                            r = resp.json()
                            if r.get("success"):
                                st.success(f"上传成功：{r.get('chunks', 0)} 个片段")
                            else:
                                st.error(f"上传失败: {r.get('error')}")
                        else:
                            st.error(f"HTTP {resp.status_code}")
                    except requests.RequestException as e:
                        st.error(f"请求失败: {e}")

        st.divider()
        st.markdown("**现有来源**")
        try:
            resp = requests.get(
                f"{API_BASE}/knowledge/sources",
                headers=_headers,
                timeout=5,
            )
            if resp.status_code == 200:
                sources = resp.json().get("sources", [])
                if not sources:
                    st.info("知识库为空")
                else:
                    for src in sources:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.code(src)
                        with col2:
                            if st.button("删除", key=f"kb-del-{src}"):
                                del_resp = requests.delete(
                                    f"{API_BASE}/knowledge/{src}",
                                    headers=_headers,
                                    timeout=10,
                                )
                                if del_resp.status_code == 200:
                                    st.success("已删除")
                                else:
                                    st.error("删除失败")
            else:
                st.error(f"获取来源失败: HTTP {resp.status_code}")
        except requests.RequestException as e:
            st.warning(f"无法连接 API：{e}")

    # ===== 健康检查 =====
    with tab_health:
        if st.button("检查健康", use_container_width=True):
            try:
                resp = requests.get(
                    f"{API_BASE}/health",
                    headers=_headers,
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", "unknown")
                    if status == "ok":
                        st.success(f"状态: {status}")
                    else:
                        st.warning(f"状态: {status}")
                    st.caption(f"版本: {data.get('version')}")
                    st.json(data.get("dependencies", {}))
                    st.json(data.get("metrics", {}))
                else:
                    st.error(f"HTTP {resp.status_code}")
            except requests.RequestException as e:
                st.error(f"无法连接 API：{e}")

        st.divider()
        st.caption(f"API 端点: `{API_BASE}`")
        st.caption(f"版本: {__version__}")

# 4. 页面标题
st.title("Cayz-agent")

# 5. 渲染历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. 处理用户输入与 Agent 回复
if prompt := st.chat_input("请输入您的指令..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考与检索..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                placeholder = st.empty()
                raw = ""
                for chunk, metadata in st.session_state.app.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        raw += chunk.content
                        placeholder.markdown(raw)
                safe_response = sanitize_text(raw)
                placeholder.markdown(safe_response)
                st.session_state.messages.append({"role": "assistant", "content": safe_response})
            except Exception as e:
                st.error(f"Agent 执行出错: {sanitize_exception(e)}")
