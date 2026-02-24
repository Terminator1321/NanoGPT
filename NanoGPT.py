import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)

        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

        self.mask: torch.Tensor
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        mask = self.mask[:T, :T]

        attn_out, _ = self.attn(
            self.ln1(x),
            self.ln1(x),
            self.ln1(x),
            attn_mask=(mask == 0)
        )

        x = x + attn_out

        x = x + self.mlp(self.ln2(x))

        return x
    
class NanoGPT(nn.Module):
    def __init__(self,vocab_size=4000,block_size=256,n_embd=384,n_head=6,n_layer=6,dropout=0.2):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)

        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
        
    def load(self, path="nanogpt_model.pt"):
        self.load_state_dict(torch.load(path))
        self.eval()
        
def generate(model, idx, max_new_tokens, stop_token_id=None,temperature=0.8, top_k=40):

    model.eval()

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -model.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        idx = torch.cat((idx, next_token), dim=1)

        if stop_token_id is not None:
            if next_token.item() == stop_token_id:
                break

    return idx

def stream_generate(model, idx, tokenizer, max_new_tokens=80,stop_token_id=None, temperature=0.7, top_k=30):

    model.eval()

    input_len = idx.shape[1]

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -model.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        idx = torch.cat((idx, next_token), dim=1)

        token_id = next_token.item()

        if stop_token_id is not None and token_id == stop_token_id:
            break

        text = tokenizer.decode([token_id])

        print(text, end="", flush=True)

    print()
    
class FeedBackTraning:
    def __init__(self, model, tokenizer, device,feedback_file="feedback_data.jsonl",lr=1e-5):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.feedback_file = feedback_file

        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=lr)
        if not os.path.exists(self.feedback_file):
            open(self.feedback_file, "w").close()

    def save_feedback(self, prompt, correct_answer):
        sample = {
            "text": f"{prompt}{correct_answer}<END>"
        }

        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")

    def train_on_feedback(self, epochs=1):
        self.model.train()

        with open(self.feedback_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) == 0:
            print("No feedback data found.")
            return

        for epoch in range(epochs):
            total_loss = 0

            for line in lines:
                data = json.loads(line)
                text = data["text"]

                ids = self.tokenizer.encode(text).ids

                if len(ids) < 2:
                    continue

                x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
                y = torch.tensor([ids[1:]], dtype=torch.long, device=self.device)

                logits, loss = self.model(x, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(lines)
            print(f"[Feedback Training] Epoch {epoch+1} | Loss {avg_loss:.4f}")

        self.model.eval()

    def save_model(self, path="finishedVersionModels/NanoGPT_feedback.pt"):
        torch.save(self.model.state_dict(), path)
        print("Feedback-trained model saved.")