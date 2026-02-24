import torch
from NanoGPT import NanoGPT as gpt
from NanoGPT import generate, stream_generate, FeedBackTraning
from tokenizers import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

def load_model(choice: int):
    model = gpt().to(device)

    match choice:
        case 1:
            path = "finishedVersionModels/NanoGPT_v1.pt"
            print("Loaded Base Model")
        case 2:
            path = "finishedVersionModels/NanoGPT_v2.pt"
            print("Loaded Chat Model")
        case 3:
            path = "finishedVersionModels/NanoGPT_v3.pt"
            print("Loaded Short Response Model")
        case 4:
            path = "finishedVersionModels/NanoGPT_v4_feedback.pt"
            print("Loaded Feedback Model")
        case _:
            path = "finishedVersionModels/NanoGPT_v3.pt"
            print("Invalid choice. Defaulting to Short Model.")

    model.load(path)
    model.eval()
    return model

current_choice = 3
model = load_model(current_choice)
feedback_system = FeedBackTraning(model, tokenizer, device)

mode = "full"
last_prompt = None
last_response = None
last_user_input = None

print("\nCommands:")
print("/1 /2 /3 /4 → switch models")
print("/a → full response mode")
print("/b → streaming mode")
print("/fix <correct answer> → manually correct last response")
print("/train_feedback → train on saved corrections")
print("/save_feedback_model → save updated model")
print("/bye → exit\n")


while True:

    user_input = input("You: ").strip()

    if user_input.lower() == "/bye":
        print("Exiting...")
        break

    if user_input in ["/1", "/2", "/3", "/4"]:
        current_choice = int(user_input[1])
        model = load_model(current_choice)
        feedback_system = FeedBackTraning(model, tokenizer, device)
        continue

    if user_input.lower() == "/a":
        mode = "full"
        print("Switched to full response mode.")
        continue

    if user_input.lower() == "/b":
        mode = "stream"
        print("Switched to streaming mode.")
        continue

    if user_input.startswith("/fix "):
        correction = user_input[5:].strip()

        if last_prompt is None:
            print("No previous response to correct.")
            continue

        feedback_system.save_feedback(last_prompt, correction)
        print("Correction saved.")
        continue

    if user_input == "/train_feedback":
        feedback_system.train_on_feedback(epochs=2)
        continue

    if user_input == "/save_feedback_model":
        feedback_system.save_model("finishedVersionModels/NanoGPT_v4_feedback.pt")
        continue

    last_user_input = user_input
    prompt = f"<USER>\n{user_input}\n<AI>\n"

    encoded = tokenizer.encode(prompt).ids
    input_len = len(encoded)

    start = torch.tensor([encoded], dtype=torch.long, device=device)
    end_token_id = tokenizer.token_to_id("<END>")

    if mode == "full":

        generated = generate(model,start,max_new_tokens=80,stop_token_id=end_token_id,temperature=0.7,top_k=30)

        new_tokens = generated[0][input_len:]
        decoded = tokenizer.decode(new_tokens.tolist())
        ai_part = decoded.replace("<END>", "").strip()

        print("AI:", ai_part)

        last_prompt = prompt
        last_response = ai_part

    else:
        print("AI: ", end="", flush=True)

        stream_generate(model,start,tokenizer,max_new_tokens=80,stop_token_id=end_token_id,temperature=0.7,top_k=30)

        print("\n(Streaming mode does not support verification)")