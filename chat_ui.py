import sys
import types
if 'xxhash' not in sys.modules:
    try:
        import xxhash
    except Exception:
        m = types.ModuleType('xxhash')
        m.xxh64_hexdigest = lambda s, seed=0: '0'*16
        m.xxh3_64_hexdigest = lambda s, seed=0: '0'*16
        m.xxh128_hexdigest = lambda s, seed=0: '0'*32
        m.xxh3_128_hexdigest = lambda s, seed=0: '0'*32
        m.xxh32_hexdigest = lambda s, seed=0: '0'*8
        sys.modules['xxhash'] = m

import base64
import os
import re
import time
import uuid

import httpx
import streamlit as st
import streamlit.components.v1 as components
import logging
import sys

# Configure 'app' logging to print to stdout so logs are visible in the Streamlit terminal
app_logger = logging.getLogger("app")
app_logger.setLevel(logging.INFO)
if not app_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    app_logger.addHandler(handler)
    app_logger.propagate = False

# Demo mode (default): run bot inside Streamlit — secrets: MONGODB_URI, GEMINI_API_KEY, etc.
# Optional: set API_BASE_URL to call a separate FastAPI server instead.
USE_REMOTE_API = bool(os.getenv("API_BASE_URL"))
API_BASE = os.getenv("API_BASE_URL", "").rstrip("/")

st.set_page_config(page_title="StayInn Booking Bot", page_icon="🏨", layout="centered")


@st.cache_resource
def _init_inline_bot() -> bool:
    """Connect MongoDB + compile LangGraph once per Streamlit process."""
    from app.config.settings import get_settings
    settings = get_settings()
    active_prov = os.getenv("LLM_PROVIDER", getattr(settings, "LLM_PROVIDER", "deepseek")).lower()
    os.environ["LLM_PROVIDER"] = active_prov
    settings.LLM_PROVIDER = active_prov

    from app.services.chat_runner import ensure_bot_ready
    from app.utils.streamlit_async import run_async

    run_async(ensure_bot_ready())
    return True


if USE_REMOTE_API:
    st.sidebar.caption(f"API mode → `{API_BASE}`")
else:
    try:
        _init_inline_bot()
        if os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud":
            st.sidebar.caption("Demo mode — bot runs in Streamlit")
    except Exception as e:
        st.error("Could not start the bot. Check Streamlit **Secrets** (or `.env` locally).")
        st.code(
            "MONGODB_URI = \"mongodb+srv://...\"\n"
            "LLM_PROVIDER = \"deepseek\"  # or \"gemini\"\n"
            "DEEPSEEK_API_KEY = \"sk-...\"\n"
            "GEMINI_API_KEY = \"...\"\n"
            "HOTEL_ID = 111111",
            language="toml",
        )
        st.caption(str(e))
        st.stop()

# ── CUSTOM PREMIUM CSS STYLING ──────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global page styling */
    .stApp {
        background: radial-gradient(circle at top left, #0e111a, #07080c);
        font-family: 'Outfit', sans-serif;
        color: #e2e8f0;
    }

    /* Gradient header */
    h1 {
        background: linear-gradient(135deg, #818cf8, #3b82f6, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        font-size: 2.6rem !important;
        margin-bottom: 0px !important;
        letter-spacing: -0.02em;
    }
    
    /* Caption styling */
    .stCaption {
        color: #94a3b8 !important;
        font-weight: 400;
        margin-top: 4px;
        margin-bottom: 24px;
    }

    /* Premium chat bubble container */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Interactive hover on chat bubbles */
    .stChatMessage:hover {
        border-color: rgba(129, 140, 248, 0.2) !important;
        box-shadow: 0 8px 32px rgba(129, 140, 248, 0.08) !important;
    }

    /* Custom scrollbar for premium look */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(129, 140, 248, 0.3);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(129, 140, 248, 0.5);
    }

    /* Sleek, premium styling for st.button */
    div.stButton > button {
        background-color: rgba(129, 140, 248, 0.08) !important;
        color: #818cf8 !important;
        border: 1px solid rgba(129, 140, 248, 0.25) !important;
        border-radius: 12px !important;
        padding: 6px 14px !important;
        font-weight: 500 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: rgba(129, 140, 248, 0.18) !important;
        border-color: #818cf8 !important;
        box-shadow: 0 4px 15px rgba(129, 140, 248, 0.25) !important;
        color: #ffffff !important;
    }
    /* Expander styling */
    .stDetails {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    /* Custom Input Bar — sticky at bottom, remounts cleanly after each send */
    .block-container {
        padding-bottom: 120px !important;
    }
    iframe[title*="custom_chat_input"] {
        position: sticky !important;
        bottom: 0 !important;
        z-index: 999 !important;
        max-width: 704px !important;
        width: 100% !important;
        margin: 0 auto !important;
        display: block !important;
        border: none !important;
        background: transparent !important;
    }
    /* Provider pill badge */
    .provider-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 8px;
    }
    .pill-deepseek {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pill-gemini {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header layout with a prominent "New Chat" button
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🏨 StayInn Booking Bot")
with col2:
    st.write("")  # Alignment spacing
    st.write("")
    if st.button("🔄 New Chat", key="new_chat_main", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.session_id = new_id
        st.query_params["session_id"] = new_id
        st.session_state.messages = []
        st.session_state.processed_msg_ids = set()
        st.session_state.pending_response = None
        st.session_state.input_key = 0
        st.rerun()

active_prov = st.session_state.get("llm_provider", os.getenv("LLM_PROVIDER", "deepseek")).lower()
prov_label = "DeepSeek Flash" if active_prov == "deepseek" else "Gemini Flash"
pill_cls = "pill-deepseek" if active_prov == "deepseek" else "pill-gemini"
icon = "🟢" if active_prov == "deepseek" else "🔵"

st.markdown(
    f'<div style="display: flex; align-items: center; margin-top: 4px; margin-bottom: 20px;">'
    f'<span style="color: #94a3b8; font-size: 0.95rem;">Talk to our AI to book your room!</span>'
    f'<span class="provider-pill {pill_cls}">{icon} {prov_label}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# Initialize session state for chat history and session ID
if "session_id" not in st.session_state:
    if "session_id" in st.query_params:
        st.session_state.session_id = st.query_params["session_id"]
    else:
        st.session_state.session_id = str(uuid.uuid4())
        st.query_params["session_id"] = st.session_state.session_id

if "messages" not in st.session_state:
    st.session_state.messages = []
    # If recovering a session, fetch history from MongoDB
    if "session_id" in st.query_params and not USE_REMOTE_API:
        try:
            from app.repositories.session_repository import get_session
            from app.utils.streamlit_async import run_async
            session_data = run_async(get_session(st.session_state.session_id))
            if session_data:
                for msg in session_data.messages:
                    st.session_state.messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
        except Exception as e:
            pass # Failsafe if DB isn't ready yet

if "processed_msg_ids" not in st.session_state:
    st.session_state.processed_msg_ids = set()
if "pending_response" not in st.session_state:
    st.session_state.pending_response = None
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Sidebar controls
with st.sidebar:
    st.title("⚙️ Controls")

    property_options = {
        "Sample Property by Sahu Hotels (988273)": "988273",
        "Sample Hotel (111111)": "111111",
    }
    selected_prop_label = st.selectbox(
        "🏨 Hotel Property",
        list(property_options.keys()),
        index=0,
        key="hotel_prop_selector"
    )
    current_selected_hid = property_options[selected_prop_label]
    st.session_state["selected_hotel_id"] = current_selected_hid

    if st.button("🔄 Start New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.session_id = new_id
        st.query_params["session_id"] = new_id
        st.session_state.messages = []
        st.session_state.processed_msg_ids = set()
        st.session_state.pending_response = None
        st.session_state.input_key = 0
        st.rerun()

    # ── LLM ENGINE CONTROLS ──
    st.divider()
    st.markdown("### 🧠 LLM Engine")
    from app.config.settings import get_settings
    settings = get_settings()

    current_env_provider = os.getenv("LLM_PROVIDER", getattr(settings, "LLM_PROVIDER", "deepseek")).lower()
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = current_env_provider if current_env_provider in ("deepseek", "gemini") else "deepseek"

    provider_labels = {
        "deepseek": f"DeepSeek ({getattr(settings, 'DEEPSEEK_MODEL', 'deepseek-flash')})",
        "gemini": f"Google Gemini ({getattr(settings, 'GEMINI_MODEL', 'gemini-2.5-flash')})",
    }

    selected_provider = st.selectbox(
        "Active Model Provider",
        options=["deepseek", "gemini"],
        index=0 if st.session_state.llm_provider == "deepseek" else 1,
        format_func=lambda k: provider_labels.get(k, k),
        help="Select which AI model powers chat and extraction.",
        key="provider_select_widget"
    )

    if selected_provider != st.session_state.llm_provider:
        st.session_state.llm_provider = selected_provider
        os.environ["LLM_PROVIDER"] = selected_provider
        settings.LLM_PROVIDER = selected_provider
        st.rerun()

    # Active provider badge and status
    if st.session_state.llm_provider == "deepseek":
        if settings.DEEPSEEK_API_KEY:
            st.markdown(
                '<div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 8px 12px; font-size: 0.82rem; color: #10b981; margin-bottom: 8px;">'
                '🟢 <b>DeepSeek Active</b><br><span style="color: #94a3b8; font-size: 0.76rem;">Model: deepseek-flash &bull; Text + Vision</span>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ DEEPSEEK_API_KEY not configured in .env")
    else:
        if settings.GEMINI_API_KEY:
            st.markdown(
                '<div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 8px 12px; font-size: 0.82rem; color: #60a5fa; margin-bottom: 8px;">'
                '🔵 <b>Gemini Active</b><br><span style="color: #94a3b8; font-size: 0.76rem;">Model: gemini-2.5-flash &bull; Multimodal</span>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ GEMINI_API_KEY not configured in .env")

    # ── LIVE FRONT DESK OPS & SMART REPLIES INSPECTOR ──
    st.divider()
    st.markdown("### 🛎️ Front Desk Ops & Tasks")
    
    # Query tasks for the current session or recent hotel tasks
    active_tasks = []
    active_tickets = []
    if not USE_REMOTE_API:
        try:
            from app.db.mongodb import hotel_tasks_collection, support_tickets_collection
            from app.config.settings import get_settings
            from app.utils.streamlit_async import run_async

            settings = get_settings()
            current_hid = str(st.session_state.get("selected_hotel_id") or "988273")
            current_sid = st.session_state.get("session_id", "")

            async def _fetch_ops():
                tasks_col = hotel_tasks_collection()
                tickets_col = support_tickets_collection()

                # Find tasks created for this session first, or recent hotel tasks
                ts = await tasks_col.find(
                    {"session_id": current_sid}
                ).sort("created_at", -1).limit(4).to_list(length=4)
                if not ts:
                    ts = await tasks_col.find(
                        {"hotel_id": current_hid}
                    ).sort("created_at", -1).limit(3).to_list(length=3)

                # Find tickets created for this session or recent
                tk = await tickets_col.find(
                    {"session_id": current_sid}
                ).sort("created_at", -1).limit(3).to_list(length=3)
                if not tk:
                    tk = await tickets_col.find(
                        {"hotel_id": int(current_hid) if current_hid.isdigit() else current_hid}
                    ).sort("created_at", -1).limit(2).to_list(length=2)

                return ts, tk

            active_tasks, active_tickets = run_async(_fetch_ops())
        except Exception:
            pass

    if active_tasks:
        for t in active_tasks:
            t_id = t.get("task_id", "TASK")
            dept = t.get("department", "FRONT_DESK")
            subtype = t.get("subtype", "GENERAL")
            prio = t.get("priority", "NORMAL")
            status = t.get("status", "OPEN")
            summary = t.get("summary", "")
            
            prio_badge = "🔴 HIGH" if prio == "HIGH" else "🟡 " + prio
            with st.expander(f"📌 {t_id} ({prio_badge})", expanded=True):
                st.markdown(f"**Dept:** `{dept}` | **Subtype:** `{subtype}`")
                st.markdown(f"**Status:** `{status}`")
                if summary:
                    st.caption(summary.replace("\n", "  \n"))
                notes = t.get("notes", [])
                if notes:
                    st.caption(f"📝 *Latest note:* {notes[-1][:120]}")
    else:
        st.caption("No operational tasks registered yet. Type a payment claim or room request to generate one!")

    st.divider()
    st.markdown("### 💬 Staff Smart Replies")
    if active_tickets:
        for tk in active_tickets:
            tk_id = tk.get("ticket_id", "TKT")
            issue = tk.get("issue_type", "GENERAL")
            reason = tk.get("reason_code", "")
            replies = tk.get("smart_replies", [])
            
            with st.expander(f"🎫 {tk_id} ({issue})", expanded=True):
                st.caption(f"Reason: `{reason}`" if reason else "")
                if replies:
                    st.markdown("**💡 AI Suggested Replies for Staff:**")
                    for r_idx, rep in enumerate(replies, 1):
                        st.info(f"**{r_idx}.** {rep}")
                else:
                    st.caption("No smart replies available for this ticket.")
    else:
        st.caption("No escalation tickets yet. When an issue occurs, 3 tailored quick replies for hotel staff will appear here.")

    # ── DEMO QUICK SCENARIOS ──
    with st.expander("⚡ Demo Quick Scenarios (Click to Copy)", expanded=False):
        st.markdown("**1. 💳 Payment via Brother's UPI:**")
        st.code("I have paid ₹5,000 from my brother's GPay, UTR is 998877665544", language="text")
        st.markdown("**2. 🏦 Direct Bank Transfer (NEFT):**")
        st.code("Transferred ₹12,000 via NEFT to hotel account, please verify", language="text")
        st.markdown("**3. 🧹 In-Stay Service Request:**")
        st.code("Please send 2 extra water bottles and towels to room 204", language="text")
        st.markdown("**4. 🚨 Escalation & Smart Replies:**")
        st.code("I have paid full amount yesterday but bot still says pending. Connect me to manager immediately!", language="text")

# Real-time text streaming generator with Markdown line break handling
def stream_text(text: str):
    # Convert all single newlines into double-space Markdown breaks
    formatted_text = text.replace("\n", "  \n")
    tokens = re.split(r"(\s+)", formatted_text)
    for token in tokens:
        if token:
            yield token
            # Natural feeling speed with short pauses on newlines/punctuation
            time.sleep(0.012 if "\n" not in token else 0.04)

def _finish_send(bot_response, booking_stage):
    st.session_state.pending_response = {
        "response": bot_response,
        "booking_stage": booking_stage,
    }
    st.session_state.input_key += 1
    st.rerun()


def _send_text_message(text: str, session_id: str, hotel_id: str | None = None) -> tuple[list, str]:
    active_prov = st.session_state.get("llm_provider", os.getenv("LLM_PROVIDER", "deepseek")).lower()
    os.environ["LLM_PROVIDER"] = active_prov
    from app.config.settings import get_settings
    get_settings().LLM_PROVIDER = active_prov

    if USE_REMOTE_API:
        payload = {"message": text, "session_id": session_id}
        if hotel_id:
            payload["hotel_id"] = hotel_id
        response = httpx.post(
            f"{API_BASE}/api/chat",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", []), data.get("booking_stage", "new")

    from app.services.chat_runner import run_chat_turn
    from app.utils.streamlit_async import run_async

    data = run_async(run_chat_turn(message=text, session_id=session_id, hotel_id=hotel_id))
    return data.get("response", []), data.get("booking_stage", "new")


def _send_media_message(
    f_bytes: bytes,
    f_name: str,
    f_type: str,
    text: str,
    session_id: str,
    hotel_id: str | None = None,
) -> tuple[list, str]:
    active_prov = st.session_state.get("llm_provider", os.getenv("LLM_PROVIDER", "deepseek")).lower()
    os.environ["LLM_PROVIDER"] = active_prov
    from app.config.settings import get_settings
    get_settings().LLM_PROVIDER = active_prov

    if USE_REMOTE_API:
        files = {"file": (f_name, f_bytes, f_type)}
        form = {"session_id": session_id}
        if hotel_id:
            form["hotel_id"] = str(hotel_id)
        if text:
            form["message"] = text
        response = httpx.post(f"{API_BASE}/api/chat/media", files=files, data=form, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        return data.get("response", []), data.get("booking_stage", "new")

    from app.services.chat_runner import run_media_turn
    from app.utils.streamlit_async import run_async

    data = run_async(
        run_media_turn(
            file_bytes=f_bytes,
            filename=f_name,
            content_type=f_type,
            session_id=session_id,
            message=text or None,
            hotel_id=hotel_id,
        )
    )
    return data.get("response", []), data.get("booking_stage", "new")


def render_cards(cards, stream=False, key_prefix=""):
    for c_idx, card in enumerate(cards):
        if card.get("text"):
            st.markdown(card["text"].replace("\n", "  \n"))
        
        if card.get("type") in ("upsell_options", "addon_catalog") and card.get("data"):
            items = card["data"]
            for item in items:
                with st.container(border=True):
                    st.markdown(f"✨ **{item.get('name')}** — ₹{item.get('price', 0):,.0f}")
                    if item.get('category'):
                        st.caption(f"Category: {item.get('category')}")
                    if item.get('description'):
                        st.markdown(f"_{item.get('description')}_")
                    custom_p = item.get("custom_pricing", [])
                    if custom_p:
                        tier_strs = [f"**{cp.get('label')}**: ₹{float(cp.get('price', 0)):,.0f}" for cp in custom_p if "label" in cp]
                        if tier_strs:
                            st.markdown("🚘 *Tier Options:* " + r" \| ".join(tier_strs))

        if card.get("buttons"):
            buttons = card.get("buttons") or []
            if buttons:
                b_cols = st.columns(min(len(buttons), 3))
                for b_idx, btn in enumerate(buttons):
                    if isinstance(btn, dict):
                        b_title = btn.get("title") or btn.get("label") or btn.get("value") or btn.get("id") or str(btn)
                        b_val = btn.get("id") or btn.get("value") or btn.get("action") or b_title
                    else:
                        b_title = str(btn)
                        b_val = str(btn)
                    with b_cols[b_idx % len(b_cols)]:
                        if st.button(b_title, key=f"{key_prefix}btn_{c_idx}_{b_idx}_{b_val}", use_container_width=True):
                            st.session_state["queued_button_click"] = (b_val, b_title)
                            st.rerun()
            if card.get("footer"):
                st.caption(f"_{card['footer']}_")

        elif card.get("type") == "image":
            url = card.get("url") or card.get("image_url") or card.get("media_url")
            caption = card.get("caption", "")
            if url:
                st.image(url, caption=caption)
            
        elif card.get("type") == "video":
            st.video(card.get("url"))
            if card.get("caption"):
                st.caption(card.get("caption"))
                
        elif card.get("type") == "audio":
            st.audio(card.get("url"))
            
        elif card.get("type") == "document":
            st.markdown(f"📄 **[Download {card.get('filename', 'Document')}]({card.get('url')})**")
            if card.get("caption"):
                st.caption(card.get("caption"))
                
        elif card.get("type") == "media_delivery":
            # For the local Streamlit UI, we resolve the media_delivery card directly to display images.
            data = card.get("data", {})
            try:
                from app.services.hotel_media_service import hotel_media_service
                from app.utils.streamlit_async import run_async
                from app.config.settings import get_settings
                
                settings = get_settings()
                card_hid = data.get("hotel_id") or card.get("hotel_id") or st.session_state.get("selected_hotel_id") or settings.HOTEL_ID
                images = run_async(hotel_media_service.find_images(
                    hotel_id=card_hid,
                    entity_type=data.get("entity_type"),
                    entity_id=data.get("entity_id"),
                    tags=data.get("tags"),
                    aliases=data.get("aliases"),
                    limit=data.get("limit", 1)
                ))
                if images:
                    for img in images:
                        url = img.get("gcs_url")
                        if url:
                            st.image(url, caption=data.get("caption") or img.get("caption") or "")
                else:
                    # Only show unavailable note if explicit photo query was made, avoid cluttering room search
                    if data.get("explicit_request") and not data.get("is_preview"):
                        from app.api.routes.whatsapp import _build_photo_unavailable_message
                        st.markdown(f"_{_build_photo_unavailable_message([card])}_")
            except Exception as e:
                import logging
                logging.error(f"Failed to load media delivery card in UI: {e}")

# Display chat messages from history
for m_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            if "file_name" in message and "file_bytes" in message:
                f_name = message["file_name"]
                f_type = message["file_type"]
                f_bytes = message["file_bytes"]
                
                if f_type.startswith("image/"):
                    st.image(f_bytes, caption=f_name, width=300)
                elif f_type.startswith("audio/"):
                    st.audio(f_bytes, format=f_type)
                else:
                    st.info(f"📄 {f_name}")
            
            if message.get("content"):
                st.markdown(str(message["content"]).replace("\n", "  \n"))
        else:
            if isinstance(message["content"], list):
                render_cards(message["content"], stream=False, key_prefix=f"hist_{m_idx}_")
            else:
                st.markdown(str(message["content"]).replace("\n", "  \n"))

# Stream pending response if there is one
if st.session_state.get("pending_response") is not None:
    pending = st.session_state.pending_response
    bot_response = pending["response"]
    booking_stage = pending.get("booking_stage")
    
    with st.chat_message("assistant"):
        if isinstance(bot_response, str):
            st.markdown(bot_response.replace("\n", "  \n"))
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
        else:
            # Structured cards (room options, quotes) — render instantly, don't stream text
            has_structured = any(
                c.get("type") in ("room_options", "quote", "payment", "upsell_options", "addon_catalog")
                for c in bot_response
                if isinstance(c, dict)
            )
            render_cards(bot_response, stream=not has_structured, key_prefix="pending_")
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
    st.session_state.pending_response = None


# Handle interactive button click if queued
if st.session_state.get("queued_button_click"):
    btn_val, btn_title = st.session_state.pop("queued_button_click")
    st.chat_message("user").markdown(btn_title)
    st.session_state.messages.append({"role": "user", "content": btn_title})

    with st.spinner("Thinking..."):
        try:
            bot_response, booking_stage = _send_text_message(
                btn_val,
                st.session_state.session_id,
                hotel_id=st.session_state.get("selected_hotel_id"),
            )
            if not bot_response:
                bot_response = [{"type": "text", "text": "Sorry, I couldn't understand that."}]
            _finish_send(bot_response, booking_stage)
        except Exception as e:
            st.error(f"Failed to get a response: {e}")

# Declare custom component
component_path = os.path.join(os.path.dirname(__file__), "app", "components", "chat_input")
custom_chat_input = components.declare_component("custom_chat_input", path=component_path)

# Render custom input (key changes after each send so the widget remounts fresh)
payload = custom_chat_input(key=f"chat_input_widget_{st.session_state.input_key}", default=None)


if payload is not None:
    msg_id = payload.get("message_id")
    
    if msg_id and msg_id not in st.session_state.processed_msg_ids:
        st.session_state.processed_msg_ids.add(msg_id)
        
        text = payload.get("text", "")
        file_data = payload.get("file")
        
        # Display user message
        if file_data:
            f_name = file_data.get("name", "attachment")
            f_type = file_data.get("type", "application/octet-stream")
            f_bytes = base64.b64decode(file_data.get("data", ""))
            
            st.session_state.messages.append({
                "role": "user",
                "content": text if text else None,
                "file_name": f_name,
                "file_type": f_type,
                "file_bytes": f_bytes
            })
            
            with st.chat_message("user"):
                if f_type.startswith("image/"):
                    try:
                        st.image(f_bytes, caption=f_name, width=300)
                    except Exception as e:
                        st.warning(f"Could not preview image: {e}")
                        st.info(f"📄 {f_name}")
                elif f_type.startswith("audio/"):
                    try:
                        st.audio(f_bytes, format=f_type)
                    except Exception:
                        st.info(f"🔊 {f_name}")
                else:
                    st.info(f"📄 {f_name}")
                if text:
                    st.markdown(text)
                    
            with st.spinner("Processing media..."):
                try:
                    bot_response, booking_stage = _send_media_message(
                        f_bytes,
                        f_name,
                        f_type,
                        text,
                        st.session_state.session_id,
                        hotel_id=st.session_state.get("selected_hotel_id"),
                    )
                    if not bot_response:
                        bot_response = [{"type": "text", "text": "Sorry, I couldn't understand that."}]
                    _finish_send(bot_response, booking_stage)
                except Exception as e:
                    st.session_state.processed_msg_ids.discard(msg_id)
                    err_msg = str(e)
                    if "Voice notes are temporarily unavailable" in err_msg or "cannot read audio" in err_msg.lower():
                        bot_response = [{
                            "type": "text",
                            "text": "🎙️ Voice notes are temporarily unavailable with DeepSeek. Please type your message or send a photo (ID card, payment receipt, room reference) — both work normally!"
                        }]
                        _finish_send(bot_response, "new")
                    else:
                        st.error(f"Failed to process media request: {e}")
                    
        elif text:
            st.chat_message("user").markdown(text)
            st.session_state.messages.append({"role": "user", "content": text})

            with st.spinner("Thinking..."):
                try:
                    bot_response, booking_stage = _send_text_message(
                        text,
                        st.session_state.session_id,
                        hotel_id=st.session_state.get("selected_hotel_id"),
                    )
                    if not bot_response:
                        bot_response = [{"type": "text", "text": "Sorry, I couldn't understand that."}]
                    _finish_send(bot_response, booking_stage)
                except Exception as e:
                    st.session_state.processed_msg_ids.discard(msg_id)
                    st.error(f"Failed to get a response: {e}")
