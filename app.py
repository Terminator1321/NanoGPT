import os
import torch
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

from NanoGPT import NanoGPT as gpt
from NanoGPT import generate, FeedBackTraning
from tokenizers import Tokenizer

from visualizer import recorder, timeline


app = Flask(__name__)
app.secret_key = "nano_gpt_secret_key"

CORS(app, supports_credentials=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

MODEL_PATHS = {
    "v1": "finishedVersionModels/NanoGPT_v1.pt",
    "v2": "finishedVersionModels/NanoGPT_v2.pt",
    "v3": "finishedVersionModels/NanoGPT_v3.pt",
    "v4": "finishedVersionModels/NanoGPT_v4_feedback.pt",
}

current_model_key = "v3"
model = None
feedback_system = None
num_blocks = 6        

def attach_visual_hooks(mdl):
    mdl.token_embedding.register_forward_hook(
        recorder.record("embedding")
    )
    for i, block in enumerate(mdl.blocks):
        block.attn.register_forward_hook(recorder.record(f"attn_{i}"))
        block.mlp.register_forward_hook(recorder.record(f"mlp_{i}"))
    mdl.ln_f.register_forward_hook(recorder.record("final_norm"))


def load_model(key):
    global model, feedback_system, current_model_key

    if key not in MODEL_PATHS:
        return False

    path = MODEL_PATHS[key]
    if not os.path.exists(path):
        return False

    model = gpt().to(device)
    model.load(path)
    model.eval()

    attach_visual_hooks(model)
    feedback_system = FeedBackTraning(model, tokenizer, device)
    current_model_key = key
    print(f"[+] Loaded model: {key}")
    return True


load_model(current_model_key)


@app.route("/")
def index():
    session["chat_history"] = []
    return render_template(
        "index.html",
        models=list(MODEL_PATHS.keys()),
        current=current_model_key,
    )


@app.route("/switch_model", methods=["POST"])
def switch_model():
    data = request.get_json()
    if not data or "model" not in data:
        return jsonify({"status": "Invalid request"}), 400
    if load_model(data["model"]):
        session["chat_history"] = []
        return jsonify({"status": f"Switched to {data['model']}"})
    return jsonify({"status": "Model not found"}), 404


@app.route("/tokenize", methods=["POST"])
def tokenize_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    raw = data["text"]
    prompt = f"<USER>\n{raw}\n<AI>\n"

    try:
        enc = tokenizer.encode(prompt)
        token_strings = [tokenizer.id_to_token(i) for i in enc.ids]
        return jsonify({
            "ids":    enc.ids,
            "tokens": token_strings,
            "count":  len(enc.ids),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"status": "Invalid request"}), 400

    user_input = data["message"]

    recorder.clear()
    timeline.clear()

    timeline.add("prompt")

    prompt = f"<USER>\n{user_input}\n<AI>\n"
    encoded = tokenizer.encode(prompt).ids
    timeline.add("tokenizer", {"tokens": encoded})

    start = torch.tensor([encoded], dtype=torch.long, device=device)
    timeline.add("positional_encoding")

    end_token_id = tokenizer.token_to_id("<END>")
    with torch.no_grad():
        generated = generate(
            model,
            start,
            max_new_tokens=120,
            stop_token_id=end_token_id,
            temperature=0.7,
            top_k=30,
        )

    timeline.add("generation_complete")

    input_len  = len(encoded)
    new_tokens = generated[0][input_len:]
    decoded    = tokenizer.decode(new_tokens.tolist())
    ai_response = decoded.replace("<END>", "").strip()

    timeline.add("output")

    history = session.get("chat_history", [])
    history.append({"role": "user", "content": user_input})
    history.append({"role": "ai",   "content": ai_response})
    session["chat_history"] = history

    return jsonify({"response": ai_response})



@app.route("/activations")
def activations():
    """Full activation data (compact stats per layer)."""
    return jsonify(recorder.data)


@app.route("/activations_summary")
def activations_summary():
    """
    Returns per-layer magnitude stats that the frontend maps to 3-D block colours.
    Shape:  { layers: [{layer, attn_mean, attn_std, mlp_mean, mlp_std}, …],
              embedding: {…}, final_norm: {…} }
    """
    try:
        layers = []
        for i in range(num_blocks):
            attn_key = f"attn_{i}"
            mlp_key  = f"mlp_{i}"
            entry: dict = {"layer": i}
            if attn_key in recorder.data:
                entry["attn_mean"] = recorder.data[attn_key]["mean"]
                entry["attn_std"]  = recorder.data[attn_key]["std"]
                entry["attn_sample"] = recorder.data[attn_key].get("sample", [])
            else:
                entry["attn_mean"] = 0.0
                entry["attn_std"]  = 0.0
                entry["attn_sample"] = []

            if mlp_key in recorder.data:
                entry["mlp_mean"] = recorder.data[mlp_key]["mean"]
                entry["mlp_std"]  = recorder.data[mlp_key]["std"]
                entry["mlp_sample"] = recorder.data[mlp_key].get("sample", [])
            else:
                entry["mlp_mean"] = 0.0
                entry["mlp_std"]  = 0.0
                entry["mlp_sample"] = []

            layers.append(entry)

        embedding   = recorder.data.get("embedding",   {})
        final_norm  = recorder.data.get("final_norm",  {})

        return jsonify({
            "layers":     layers,
            "embedding":  {k: v for k, v in embedding.items()  if k != "sample"},
            "final_norm": {k: v for k, v in final_norm.items() if k != "sample"},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/timeline")
def get_timeline():
    return jsonify(timeline.timeline)



@app.route("/chat_history")
def chat_history():
    return jsonify(session.get("chat_history", []))



@app.route("/fix", methods=["POST"])
def fix():
    data = request.get_json()
    if not data or "correction" not in data:
        return jsonify({"status": "Invalid request"}), 400
    history = session.get("chat_history", [])
    if len(history) < 2:
        return jsonify({"status": "Nothing to fix"}), 400
    last_user_input = history[-2]["content"]
    prompt = f"<USER>\n{last_user_input}\n<AI>\n"
    feedback_system.save_feedback(prompt, data["correction"])#type: ignore
    return jsonify({"status": "Correction saved."})


@app.route("/train", methods=["POST"])
def train():
    feedback_system.train_on_feedback(epochs=2)#type: ignore
    return jsonify({"status": "Training complete."})


@app.route("/save", methods=["POST"])
def save():
    path = MODEL_PATHS["v4"]
    feedback_system.save_model(path)#type: ignore
    load_model("v4")
    return jsonify({"status": "Saved and reloaded v4."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)