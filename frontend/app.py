import base64
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import quote, unquote

import requests
import streamlit as st
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=APP_DIR.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("IntelliVaultUI")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8010")
LOGO_PATH = APP_DIR / "assets" / "logo.png"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="IntelliVault Blueprints",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* Hide Streamlit toolbar/header chrome */
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stBaseButton-header"],
[data-testid="stBaseButton-headerNoPadding"],
footer { display: none !important; }

/* Hide avatars */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"],
.stChatMessage > div:first-child,
[data-testid="stChatMessageAvatarContainer"] {
    display: none !important;
}

/* Header Box (card with logo inside) */
.bot-header-box {
    border: 2px solid #E5E7EB;
    border-radius: 28px;
    padding: 20px 28px 22px;
    background-color: #FFFFFF;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 28px;
}
.bot-header-logo { height: 64px; display: block; margin-bottom: 6px; }

.bot-header-title {
    font-size: 44px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 6px;
}

.bot-header-subtitle {
    font-size: 14px;
    font-weight: 500;
    color: #6B7280;
}

/* User Bubble LEFT */
.user-bubble {
    background: #1E73E8;
    color: white;
    padding: 14px 18px;
    border-radius: 16px;
    width: fit-content;
    max-width: 75%;
    margin-right: auto;
    margin-bottom: 15px;
    font-weight: 500;
    font-size: 17px;
    box-shadow: 0 4px 10px rgba(30,115,232,0.25);
}

/* AI Bubble RIGHT — only style containers that hold REAL rendered text/markdown.
   Widget labels and button text are excluded so the domain screen and Back button
   are not turned into bubbles. */
[data-testid="stMarkdownContainer"]:not(:has(.user-bubble)):not(:has(.bot-header-box)):not(:has(.export-btn-wrap)):not([data-testid="stWidgetLabel"] *):not(.stButton *):not([role="dialog"] *):not([class*="st-key-activity"] *):not([data-testid="stAlert"] *):not([data-testid="stToast"] *):has(p, ul, ol, pre, blockquote, table, h1, h2, h3, h4) {
    background: #F3F4F6;
    padding: 14px 18px;
    border-radius: 16px;
    width: fit-content;
    max-width: 75%;
    margin-left: auto;
    margin-bottom: 15px;
    font-weight: 500;
    font-size: 17px;
    color: #111827;
}

/* Export button — right-aligned rectangle with arrow */
.export-btn-wrap {
    display: flex;
    justify-content: flex-end;
    margin-top: 8px;
    margin-bottom: 4px;
}
.export-btn-wrap a {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: #ffffff;
    color: #1E73E8 !important;
    border: 2px solid #1E73E8;
    border-radius: 6px;
    padding: 7px 18px;
    font-size: 14px;
    font-weight: 700;
    text-decoration: none !important;
    letter-spacing: 0.3px;
    transition: background 0.15s, color 0.15s;
}
.export-btn-wrap a:hover {
    background: #1E73E8;
    color: #ffffff !important;
}
.export-arrow {
    font-size: 18px;
    font-weight: 900;
}

/* Empty-state hero */
.hero { text-align: center; margin: 8px 0 4px; }
.hero-icon {
    width: 110px; height: 110px; margin: 0 auto 16px; border-radius: 50%;
    background: #F5F8FF; display: flex; align-items: center; justify-content: center;
}
.hero-icon-mid {
    width: 84px; height: 84px; border-radius: 50%; background: #E6EEFC;
    display: flex; align-items: center; justify-content: center;
}
.hero-icon-sq {
    width: 44px; height: 44px; border: 3px solid #3B82F6; border-radius: 14px;
    display: flex; align-items: center; justify-content: center; box-sizing: border-box;
}
.hero-icon-sq::after { content: ""; width: 10px; height: 10px; border-radius: 50%; background: #3B82F6; }
.hero-title { font-size: 28px; font-weight: 800; color: #111827; margin-bottom: 8px; }
.hero-sub { font-size: 16px; color: #6B7280; }
/* Chat input: soft grey pill */
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background: #F3F4F6 !important; }
[data-testid="stChatInput"] > div { border: none !important; border-radius: 16px !important; }

/* New Chat button: white rounded outline, pinned just above the input bar */
.st-key-new_chat_btn {
    position: sticky; bottom: 6rem; z-index: 5;
    display: flex; justify-content: center; padding: 6px 0;
}
.st-key-new_chat_btn button {
    background: #ffffff; color: #111827; border: 1px solid #D1D5DB;
    border-radius: 12px; padding: 8px 22px; font-size: 16px; font-weight: 500;
}
.st-key-new_chat_btn button:hover { border-color: #3B82F6; color: #3B82F6; background: #ffffff; }

/* Upload dialog title: one line, bold */
[role="dialog"] h2, [role="dialog"] [data-testid="stHeading"] { font-weight: 800 !important; white-space: nowrap; }

div[data-testid="stExpander"] {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- API ----------------
def api(method: str, path: str, timeout: int = 30, **kw) -> dict | list:
    r = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kw)
    r.raise_for_status()
    return r.json()


def error_text(e: Exception) -> str:
    """Backend's `detail` message when there is one, else the raw error."""
    if isinstance(e, requests.HTTPError):
        try:
            return e.response.json().get("detail", str(e))
        except ValueError:
            pass
    return str(e)


@st.cache_data(show_spinner=False, ttl=300)
def fetch_domains() -> list[str]:
    return api("get", "/domains", timeout=10)


# ---------------- UI HELPERS ----------------
def notify(msg: str, icon: str = "✅") -> None:
    """Queue a toast; it is shown on the next run, so it survives st.rerun()."""
    st.session_state.toast = (msg, icon)


def slug_of(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def done_and_close(domain: str, msg: str) -> None:
    """After an upload/new project: keep the dialog open with a spinner until indexing
    finishes, then close it and toast the result."""
    fetch_domains.clear()
    st.success(msg)
    slug, state = slug_of(domain), None
    st.caption(f"Indexing '{domain}'... you can watch each file below.")
    board = st.empty()
    while state not in ("done", "error"):
        time.sleep(2)
        try:
            entry = api("get", "/index_status", timeout=5).get(slug) or {}
        except Exception as e:
            log.error("Index status check failed: %s", e)
            entry, state = {}, "error"
        state = entry.get("state", state)
        with board.container():
            render_progress(entry.get("files", {}), entry.get("diff"))
    if state == "done":
        notify(f"'{domain}' is indexed and ready.")
    else:
        notify(f"Indexing '{domain}' failed. Check the log file and try again.", "⚠️")
    st.rerun()


def render_progress(files: dict, diff: dict | None = None) -> None:
    """One bar per PDF (+ one for the v1/v2 comparison), from the backend's /index_status."""
    if not files:
        st.progress(0.0, text="Starting...")
        return
    n_done = sum(f["stage"].startswith("done") for f in files.values())
    st.progress(n_done / len(files), text=f"{n_done}/{len(files)} files searchable")
    for f in [*files.values(), *([diff] if diff else [])]:
        if f["stage"].startswith("error"):
            st.error(f"{f['file']}: {f['stage']}")
            continue
        total, done = f.get("total") or 0, f.get("done") or 0
        frac = 1.0 if f["stage"].startswith("done") else (done / total if total else 0.0)
        pages = f" {done}/{total} pages" if total and not f["stage"].startswith("done") else ""
        secs = f["secs"] if f.get("secs") is not None else (round(time.time() - f["t0"]) if f.get("t0") else 0)
        st.progress(min(frac, 1.0), text=f"{f['file']} · {f['stage']}{pages} · {secs}s")


@st.fragment(run_every=2)
def activity_panel(only: str | None = None) -> None:
    """Background work and unfinished projects, on the start and chat screens. `only` = one project."""
    try:
        status, unfinished = api("get", "/index_status", timeout=5), api("get", "/unfinished", timeout=5)
        labels = {slug_of(d): d for d in fetch_domains()}
    except Exception as e:
        log.error("Index status check failed: %s", e)
        return
    watched = st.session_state.setdefault("watched", set())
    for slug, entry in status.items():
        name = labels.get(slug, slug)
        if only and name != only:
            continue
        if entry["state"] == "indexing":
            watched.add(slug)
            with st.container(key=f"activity_{slug}"):
                st.caption(f"Working on '{name}'")
                render_progress(entry.get("files", {}), entry.get("diff"))
        elif entry["state"] == "error":
            with st.container(key=f"activity_{slug}"):
                st.caption(f"Problem in '{name}'")
                render_progress(entry.get("files", {}), entry.get("diff"))
        elif slug in watched:
            st.success(f"'{name}' is ready.")
    for name in unfinished:
        if only and name != only:
            continue
        with st.container(key=f"activity_unfinished_{slug_of(name)}"):
            st.caption(f"'{name}' is not finished. Continuing may use Gemini.")
            if st.button("Continue indexing", key=f"continue_{slug_of(name)}"):
                try:
                    api("post", "/continue", timeout=30, data={"domain": name})
                except Exception as e:
                    st.error(error_text(e))


def browse_folder() -> None:
    """Native Windows folder picker (the app runs on this PC); fills the path box."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # otherwise it opens behind the browser
    chosen = filedialog.askdirectory(parent=root, title="Select project folder")
    root.destroy()
    if chosen:
        st.session_state.np_path = str(Path(chosen))


def export_link(data_uri: str, filename: str) -> None:
    st.markdown(
        f'<div class="export-btn-wrap"><a href="{data_uri}" download="{filename}">'
        f'Export <span class="export-arrow">&#x27F6;</span></a></div>',
        unsafe_allow_html=True,
    )


def render_logo_and_header():
    logo_html = ""
    if LOGO_PATH.exists():
        logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        logo_html = f'<img class="bot-header-logo" src="data:image/png;base64,{logo_b64}">'
    st.markdown(f"""
<div class="bot-header-box">
    {logo_html}
    <div class="bot-header-title">
        Intelli<span style="color:#1E73E8;">Vault</span>
    </div>
    <div class="bot-header-subtitle">
        Blueprints. Sheets. Instant Answers
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- DIALOGS ----------------
KINDS = {"Blueprint": "blueprint", "Other document": "other"}


def kind_choice(key: str) -> str:
    """Blueprint / Other document radio; returns the backend's name for the choice."""
    label = st.radio("What are these files?", list(KINDS), horizontal=True, key=key)
    if KINDS[label] == "other":
        st.caption("Contracts, quotes, manuals, photos. Scanned pages and images are read by AI and marked in answers.")
    return KINDS[label]


@st.dialog("Upload documents", width="large")
def upload_dialog(domains: list[str]) -> None:
    tab_files, tab_project = st.tabs(["Add files", "Add project"])

    with tab_files:
        kind = kind_choice("up_kind")
        folder = st.selectbox("Project", domains, index=None, placeholder="Choose a project...", key="up_existing")
        files = st.file_uploader(
            "Select files", type=["pdf"] if kind == "blueprint" else ["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True, key=f"up_files_{kind}",
        )
        if st.button("Upload & index", type="primary", disabled=not (folder and files), key="up_files_btn"):
            try:
                with st.spinner("Uploading..."):
                    res = api("post", "/upload", timeout=300, data={"domain": folder, "kind": kind},
                              files=[("files", (f.name, f.getvalue())) for f in files])
            except Exception as e:
                log.error("Upload failed: %s", e)
                st.error(f"Upload failed: {error_text(e)}")
            else:
                msg = f"Uploaded {len(res['saved'])} file(s) to '{res['domain']}'."
                if res["skipped"]:
                    msg += " Skipped (unsupported type): " + ", ".join(res["skipped"])
                done_and_close(res["domain"], msg)

    with tab_project:
        name = st.text_input("Project name", placeholder="e.g. Dupage Community Center Project", key="np_name").strip()
        kind = kind_choice("np_kind")
        source = st.radio("Source", ["Upload as ZIP", "Path on this PC"], horizontal=True, key="np_source")
        zip_up, path = None, ""
        if source == "Upload as ZIP":
            zip_up = st.file_uploader("Project ZIP (subfolders are kept)", type=["zip"], key="np_zip")
        else:
            col_path, col_btn = st.columns([5, 1], vertical_alignment="bottom")
            path = col_path.text_input("Folder path", placeholder=r"C:\Projects\Dupage Community Center Project",
                                       key="np_path").strip()
            col_btn.button("Browse", key="np_browse", on_click=browse_folder, width="stretch")
        if st.button("Create project", type="primary", disabled=not (name and (zip_up or path)), key="np_btn"):
            try:
                with st.spinner("Adding project..."):
                    res = api("post", "/add_project", timeout=600, data={"name": name, "path": path, "kind": kind},
                              files={"zip_file": (zip_up.name, zip_up.getvalue())} if zip_up else None)
            except Exception as e:
                log.error("Add project failed: %s", e)
                st.error(f"Could not add project: {error_text(e)}")
            else:
                msg = f"Project '{res['domain']}' created with {res['saved']} file(s)."
                done_and_close(res["domain"], msg)


@st.dialog("Delete project")
def delete_dialog(domains: list[str]) -> None:
    target = st.selectbox("Project to delete", domains, index=None, placeholder="Choose a project...", key="del_domain")
    sure = st.checkbox("I understand this permanently deletes the project's files and index", key="del_confirm")
    if st.button("Delete", type="primary", disabled=not (target and sure), key="del_btn"):
        try:
            with st.spinner("Deleting..."):
                api("delete", f"/domains/{quote(target, safe='')}", timeout=60)
        except Exception as e:
            log.error("Delete failed: %s", e)
            st.error(f"Delete failed: {error_text(e)}")
        else:
            fetch_domains.clear()
            notify(f"Project '{target}' deleted.", "🗑️")
            st.rerun()


def go_back():
    log.info("Back pressed, leaving domain=%s", st.session_state.domain)
    st.session_state.domain = None
    st.session_state.messages = []


def new_chat():
    log.info("New chat pressed, staying in domain=%s", st.session_state.domain)
    st.session_state.messages = []


# ---------------- SESSION STATE ----------------
st.session_state.setdefault("domain", None)
st.session_state.setdefault("messages", [])
if toast := st.session_state.pop("toast", None):
    st.toast(toast[0], icon=toast[1])


# ================= SCREEN 1: DOMAIN SELECTION =================
if st.session_state.domain is None:
    render_logo_and_header()

    try:
        domains = fetch_domains()
    except Exception as e:
        log.error("Could not reach backend at %s: %s", BACKEND_URL, e)
        fetch_domains.clear()
        st.error(f"Could not reach the backend at {BACKEND_URL}. Is it running?")
        st.stop()

    choice = st.selectbox("Select your project", domains, index=None, placeholder="Choose a project...")
    if st.button("Continue", type="primary", disabled=choice is None):
        log.info("Domain selected: %s", choice)
        st.session_state.domain = choice
        st.session_state.messages = []
        st.rerun()

    col_up, col_del = st.columns(2)
    if col_up.button("Upload documents", width="stretch"):
        upload_dialog(domains)
    if col_del.button("Delete project", width="stretch", disabled=not domains):
        delete_dialog(domains)
    activity_panel()
    st.stop()


# ================= SCREEN 2: CHAT =================
st.button("← Back", on_click=go_back)

render_logo_and_header()
activity_panel(st.session_state.domain)

if not st.session_state.messages:
    st.markdown(
        """
<div class="hero">
    <div class="hero-icon"><div class="hero-icon-mid"><div class="hero-icon-sq"></div></div></div>
    <div class="hero-title">Ask the vault</div>
    <div class="hero-sub">Ask about the drawings, or say "show A101"</div>
</div>
""",
        unsafe_allow_html=True,
    )

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        continue

    st.markdown(msg["content"])

    for card in msg.get("images", []):
        st.image(f"{BACKEND_URL}{card['src']}", caption=card["label"], width="stretch")
        export_link(f"{BACKEND_URL}{card['export']}", card["filename"])

    for path in msg.get("pdfs", []):
        # Whole sets are ~50 MB: let the browser load them from the local backend, not a data URI.
        name, url = unquote(Path(path.split("#")[0]).name), f"{BACKEND_URL}{path}"  # "#page=3" opens that page
        with st.expander(f"📄 {name}", expanded=False):
            st.markdown(f'<iframe src="{url}" width="100%" height="600"></iframe>', unsafe_allow_html=True)
            export_link(url, name)

# ---------------- GENERATE RESPONSE IF THINKING ----------------
messages = st.session_state.messages
if messages and messages[-1]["role"] == "assistant" and messages[-1]["content"] == "Thinking...":
    try:
        data = api("post", "/chat", timeout=180, json={
            "domain": st.session_state.domain,
            "message": messages[-2]["content"],
            "history": [{"role": m["role"], "content": m["content"]} for m in messages[:-2]],
        })
        messages[-1].update(content=data["answer"], images=data.get("images", []), pdfs=data.get("pdfs", []))
    except Exception as e:
        log.error("Chat request failed (domain=%s): %s", st.session_state.domain, e)
        messages[-1]["content"] = f"Error: {error_text(e)}"
    st.rerun()

# ---------------- NEW CHAT BUTTON (above the input bar, only after first query) ----------------
if messages:
    st.button("+ New Chat", key="new_chat_btn", on_click=new_chat)

# ---------------- CHAT INPUT ----------------
if question := st.chat_input("Ask a question, or 'show A101'..."):
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": "Thinking..."})
    st.rerun()
