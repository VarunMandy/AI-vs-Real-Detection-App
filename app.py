"""
Real or AI? — Human vs. Model face detection game (Streamlit)
"""

import os
import glob
import json
import random

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

N_ROUNDS = 10

st.set_page_config(page_title="Real or AI? Human vs. Model", page_icon="🕵️", layout="centered")


# ------------------------------------------------------------
# Model + data (loaded once per server, shared across sessions)
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model…")
def load_everything():
    with open("config.json") as f:
        cfg = json.load(f)
    model = tf.keras.models.load_model("best_model.keras")
    scale = float(cfg.get("input_scale", 1.0))
    size = int(cfg.get("image_size", 128))

    items = []
    for label, sub in [(1, "real"), (0, "ai")]:
        for p in sorted(glob.glob(os.path.join("test_images", sub, "*"))):
            if p.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((p, label))
    if not items:
        raise RuntimeError("No images found in test_images/real or test_images/ai")

    def preprocess(path):
        # Same as training: resize to 128x128, scale to [0,1] (x255 for MobileNet models)
        img = Image.open(path).convert("RGB").resize((size, size))
        return np.asarray(img, dtype="float32") / 255.0 * scale

    # Pre-compute model scores for every image so gameplay is instant
    scores = model.predict(np.stack([preprocess(p) for p, _ in items]),
                           batch_size=32, verbose=0).ravel()
    return cfg, items, scores


CFG, ITEMS, SCORES = load_everything()


def label_name(y):
    return "Real 📷" if y == 1 else "AI-generated 🤖"


# ------------------------------------------------------------
# Game state + callbacks
# ------------------------------------------------------------
def start_game():
    st.session_state.order = random.sample(range(len(ITEMS)), min(N_ROUNDS, len(ITEMS)))
    st.session_state.round = 0
    st.session_state.human = 0
    st.session_state.model = 0
    st.session_state.answered = False
    st.session_state.done = False
    st.session_state.history = []
    st.session_state.celebrate = False


def make_guess(choice):
    s = st.session_state
    if s.answered or s.done:
        return
    idx = s.order[s.round]
    truth = ITEMS[idx][1]
    score = float(SCORES[idx])
    model_pred = int(score >= 0.5)
    entry = {"truth": truth, "guess": choice, "model_pred": model_pred, "score": score,
             "human_ok": choice == truth, "model_ok": model_pred == truth}
    s.history.append(entry)
    s.human += int(entry["human_ok"])
    s.model += int(entry["model_ok"])
    s.answered = True


def next_round():
    s = st.session_state
    if not s.answered or s.done:
        return
    if s.round == len(s.order) - 1:
        s.done = True
        s.celebrate = s.human > s.model
    else:
        s.round += 1
        s.answered = False


if "order" not in st.session_state:
    start_game()
S = st.session_state


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🕵️ Real or AI?")
st.markdown(
    f"Look at each face and decide whether it's a **real photo** or **AI-generated**. "
    f"After you guess, the model reveals its answer. **{N_ROUNDS} rounds**: highest score wins!"
)
st.caption(f"Opponent: **{CFG.get('model_name', 'CNN')}** "
           f"(test accuracy {CFG.get('test_acc', 0):.0%})")

# Scoreboard
answered = len(S.history)
total = len(S.order)
c1, c2, c3 = st.columns(3)
c1.metric("Round", f"{min(S.round + 1, total)} / {total}")
c2.metric("🧑 You", f"{S.human} / {answered}",
          f"{100 * S.human / answered:.0f}%" if answered else None, delta_color="off")
c3.metric("🤖 Model", f"{S.model} / {answered}",
          f"{100 * S.model / answered:.0f}%" if answered else None, delta_color="off")

st.divider()

if not S.done:
    # ---- Current round ----
    path = ITEMS[S.order[S.round]][0]
    left, mid, right = st.columns([1, 6, 1])
    with mid:
        st.image(path, width=420)

    b1, b2 = st.columns(2)
    b1.button("📷 Real", on_click=make_guess, args=(1,), disabled=S.answered,
              use_container_width=True, type="primary")
    b2.button("🤖 AI-generated", on_click=make_guess, args=(0,), disabled=S.answered,
              use_container_width=True)

    if not S.answered:
        st.info("🤔 Is this face **real** or **AI-generated**?")
    else:
        e = S.history[-1]
        conf = e["score"] if e["model_pred"] == 1 else 1 - e["score"]
        msg = (f"This face is **{label_name(e['truth'])}**.  \n"
               f"You said: **{label_name(e['guess'])}** {'✓' if e['human_ok'] else '✗'}  \n"
               f"Model said: **{label_name(e['model_pred'])}** ({conf:.0%} sure, "
               f"score {e['score']:.2f}) {'✓' if e['model_ok'] else '✗'}")
        if e["human_ok"] and not e["model_ok"]:
            msg += "  \n\n🏆 **You beat the model on this one!**"
        elif e["model_ok"] and not e["human_ok"]:
            msg += "  \n\n🤖 **The model got this one.**"
        elif not e["human_ok"] and not e["model_ok"]:
            msg += "  \n\n😵 **This one fooled you both!**"

        if e["human_ok"]:
            st.success("✅ **Correct!**  \n" + msg)
        else:
            st.error("❌ **Not quite!**  \n" + msg)

        last = S.round == total - 1
        st.button("🏁 See final results" if last else "Next image ➡️",
                  on_click=next_round, use_container_width=True)

else:
    # ---- Final summary ----
    n = len(S.history)
    if S.human > S.model:
        st.header("🎉 You win!")
        st.write("Humans 1, machines 0. Nice eye!")
    elif S.model > S.human:
        st.header("🤖 The model wins!")
        st.write("The machine spotted more fakes this time. Rematch?")
    else:
        st.header("🤝 It's a tie!")
        st.write("Evenly matched. Try another round.")

    if S.celebrate:
        st.balloons()
        S.celebrate = False   # only once

    st.markdown(f"### 🧑 You: {S.human}/{n} ({100 * S.human / n:.0f}%) &nbsp;|&nbsp; "
                f"🤖 Model: {S.model}/{n} ({100 * S.model / n:.0f}%)")

    st.table([
        {"#": i + 1,
         "Truth": "Real" if e["truth"] == 1 else "AI",
         "You": f"{'Real' if e['guess'] == 1 else 'AI'} {'✅' if e['human_ok'] else '❌'}",
         "Model (score)": f"{'Real' if e['model_pred'] == 1 else 'AI'} "
                          f"({e['score']:.2f}) {'✅' if e['model_ok'] else '❌'}"}
        for i, e in enumerate(S.history)
    ])

st.divider()
st.button("🔄 New game", on_click=start_game)
