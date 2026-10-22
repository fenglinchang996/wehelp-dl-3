import json
import time
from math import e
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset

from utils import get_device


def parser(file: Path) -> list[str] | None:
    with open(file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        parsed_data = [p for item in raw_data for p in item["paragraphs"]]
        return parsed_data


class Tokenizer:
    EOS = "<EOS>"

    def __init__(self, corpus: list[str]) -> None:
        preprocessed = [list(p) for p in corpus]
        all_words = sorted({t for tokens in preprocessed for t in tokens})
        all_words.append(self.EOS)
        self.id_to_word = {i: w for i, w in enumerate(all_words)}
        self.word_to_id = {w: i for i, w in enumerate(all_words)}
        self.vocab_size = len(self.id_to_word)

    def encode(self, text: str, add_eos=False) -> list[int]:
        words = [w for w in text]
        if add_eos:
            words.append(self.EOS)
        tokens = [self.word_to_id[w] for w in words]
        return tokens

    def decode(self, tokens: list[int]) -> list[str]:
        words = [self.id_to_word[t] for t in tokens]
        return words


CONTEXT_LENGTH = 128


class CustomDataset(Dataset):
    def __init__(
        self,
        parsed_data: list[str],
        tokenizer: Tokenizer,
        max_length: int = CONTEXT_LENGTH,
        stride: int = 16,
    ) -> None:
        super().__init__()
        input_tokens: list[list[int]] = []
        target_tokens: list[list[int]] = []
        tokens = [t for p in parsed_data for t in tokenizer.encode(p, add_eos=True)]
        for i in range(0, len(tokens) - max_length, stride):
            input_tokens.append(tokens[i : i + max_length])
            target_tokens.append(tokens[i + 1 : i + max_length + 1])
        self.input_tokens = torch.tensor(input_tokens)
        self.target_tokens = torch.tensor(target_tokens)

    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, idx):
        return self.input_tokens[idx], self.target_tokens[idx]


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length=CONTEXT_LENGTH,
        emb_dim=128,
        nhead=4,
        num_layers=4,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.transformers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                emb_dim,
                nhead,
                activation=nn.functional.gelu,
                dim_feedforward=emb_dim * 4,
                batch_first=True,
            ),
            num_layers,
        )
        self.out_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, input: Tensor):
        _, seq_len = input.shape
        tok_embeds = self.tok_emb(input)
        pos_embeds = self.pos_emb(torch.arange(seq_len).to(input.device))
        x = tok_embeds + pos_embeds
        x = self.transformers(
            x,
            mask=nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=input.device
            ),
            is_causal=True,
        )
        logits = self.out_head(x)
        return logits


EPOCHS = 20
BATCH_SIZE = 64
WEIGHTS_PATH = Path("gpt.pt")


def train(
    parsed_data: list[str],
    tokenizer: Tokenizer,
    device: torch.device,
):
    dataset = CustomDataset(parsed_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    model = GPT(tokenizer.vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    criteria = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for idx, (batch_X, batch_e) in enumerate(dataloader):
            optimizer.zero_grad()
            batch_X = batch_X.to(device)
            batch_e = batch_e.to(device)
            logits = model(batch_X)
            loss = criteria(logits.flatten(0, 1), batch_e.flatten())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(
                f"\repoch {epoch + 1} batch {idx + 1} done",
                flush=True,
                end="",
            )
        avg_loss = running_loss / len(dataloader)
        print(f" | epoch {epoch + 1} loss: {avg_loss} perplexity: {e ** avg_loss}")
    torch.save(model.state_dict(), WEIGHTS_PATH)
    return model


def main():
    device = get_device()
    parsed_data = parser(Path("data/classical_chinese/data.json"))
    if not parsed_data:
        return
    tokenizer = Tokenizer(parsed_data)

    model = None

    if False:
        start_time = time.time()
        model = train(parsed_data, tokenizer, device)
        end_time = time.time()
        print(f"duration: {end_time - start_time}")

    if not model:
        model = GPT(tokenizer.vocab_size).to(device)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

    MAX_NEW_TOKENS = 300
    TOP_K = 10

    model.eval()
    with torch.inference_mode():
        while True:
            prompt = input("輸入古文提示詞（輸入 exit 或 quit 退出）：").strip()
            if prompt == "exit" or prompt == "quit":
                break
            if not prompt or any(w not in tokenizer.word_to_id for w in prompt):
                continue
            encoded = torch.tensor(tokenizer.encode(prompt)).to(device).unsqueeze(0)
            for _ in range(MAX_NEW_TOKENS):
                encoded_cond = encoded[:, -CONTEXT_LENGTH:]
                logits = model(encoded_cond)
                logits = logits[:, -1, :]
                top_logits, _ = torch.topk(logits, TOP_K)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )
                probas = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probas, num_samples=1)
                if Tokenizer.EOS in tokenizer.decode(next_token.flatten().tolist()):
                    break
                encoded = torch.cat((encoded, next_token), dim=1)
            print(
                "模型生成古文：", "".join(tokenizer.decode(encoded.flatten().tolist()))
            )


if __name__ == "__main__":
    main()
