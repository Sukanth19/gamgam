"""gamgam chat UI.

    uv run streamlit run streamlit_app.py

Only display logic lives here; app/chat.py does the work.
"""

import streamlit as st

from app import db
from app.chat import HELP, Assistant

st.set_page_config(page_title="gamgam", page_icon="🎮", layout="centered")


@st.cache_resource(show_spinner="Loading models and indexes…")
def get_assistant() -> Assistant:
    # cache_resource: built once per server process, shared across reruns.
    # (Streamlit re-runs this whole script on every interaction.)
    return Assistant()


assistant = get_assistant()
state = st.session_state
state.setdefault("messages", [{"role": "assistant", "text": HELP, "sources": []}])
state.setdefault("last_game", None)

# --- sidebar: taste profile ---------------------------------------------------
with st.sidebar:
    st.header("Your games")
    assistant.explain = st.toggle("Explain recommendations (Gemini)", value=True)
    conn = db.connect()
    try:
        history = db.get_history(conn)
        for h in history:
            col1, col2 = st.columns([5, 1])
            col1.write(("👍 " if h["verdict"] == "liked" else "👎 ") + h["name"])
            if col2.button("✕", key=f"rm-{h['game_id']}", help="remove"):
                db.remove_verdict(conn, h["game_id"])
                st.rerun()
        if not history:
            st.caption("No games rated yet.")

        with st.form("rate", clear_on_submit=True):
            name = st.text_input("Add a game", placeholder="e.g. Hollow Knight")
            c1, c2 = st.columns(2)
            liked = c1.form_submit_button("👍 Liked")
            disliked = c2.form_submit_button("👎 Disliked")
        if (liked or disliked) and name.strip():
            game = db.find_game_by_name(conn, name.strip())
            if game:
                db.set_verdict(conn, game["id"], "liked" if liked else "disliked")
                st.rerun()
            else:
                st.warning(f"No game matching “{name}”.")

        st.header("Spoiler protection")
        st.caption("Lore answers only use what you've seen so far.")
        from app.progress import milestones_for
        from app.router import LORE_GAMES
        for name in LORE_GAMES:
            game = db.find_game_by_name(conn, name)
            if game is None:
                continue
            labels = ["Off (show everything)"] + [m.label for m in milestones_for(name)]
            current = db.get_progress(conn, game["id"])
            choice = st.selectbox(name, labels, index=0 if current is None else current + 1,
                                  key=f"progress-{game['id']}-{current}")
            picked = None if choice == labels[0] else labels.index(choice) - 1
            if picked != current:
                db.set_progress(conn, game["id"], picked)
                st.rerun()
    finally:
        conn.close()
    if state.last_game:
        st.caption(f"Talking about: **{state.last_game}**")


# --- chat ---------------------------------------------------------------------
def render_sources(sources: list[dict]):
    if not sources:
        return
    normal = [s for s in sources if not s["spoiler"]]
    spoilers = [s for s in sources if s["spoiler"]]

    def fmt(i, s):
        where = s["title"] + (f" › {s['section']}" if s["section"] else "")
        return f"[{i}] [{where}]({s['url']})"

    numbered = list(enumerate(sources, 1))
    st.caption("Sources: " + " · ".join(fmt(i, s) for i, s in numbered if s in normal)
               if normal else "Sources:")
    if spoilers:
        with st.expander(f"⚠️ {len(spoilers)} source(s) contain spoilers"):
            for i, s in numbered:
                if s in spoilers:
                    st.markdown(fmt(i, s))
    st.caption("Wiki text is CC BY-SA / CC BY — see the linked pages for authors.")


for msg in state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        render_sources(msg.get("sources", []))

if prompt := st.chat_input("Ask for recommendations, similar games, or lore…"):
    state.messages.append({"role": "user", "text": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking… (free-tier Gemini can take a little while)"):
            reply = assistant.handle(prompt, state.last_game)
        st.markdown(reply.text)
        render_sources(reply.sources)
    if reply.game:
        state.last_game = reply.game
    state.messages.append({"role": "assistant", "text": reply.text, "sources": reply.sources})
    st.rerun()   # refresh the sidebar (a like/dislike may have changed it)
