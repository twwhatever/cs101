import tiktoken
import torch

class Generator():
    def __init__(self, gpt, tokenizer=None):
        self._gpt = gpt
        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = tiktoken.get_encoding("gpt2")
        self._token_ids = []

    def encode(self, text):
        self._token_ids.extend(
            self._tokenizer.encode(
                text,
                allowed_special={"<|endoftext|>"},
            )
        )

    def _consume_tensor(self):
        tensor = torch.tensor(self._token_ids).unsqueeze(0)
        self._token_ids = []
        return tensor

    def generate(self, max_new_tokens):
        idx = self._consume_tensor()

        for _ in range(max_new_tokens):

            # Compute logits given the tokens in the context.
            idx_cond = idx[:, -self._gpt.context_length():]
            with torch.no_grad():
                # logits: (batch, n_token, vocab size)
                logits = self._gpt(idx_cond)

            # Focus on the last token.
            logits = logits[:, -1, :]

            # probs: (batch, vocab size)
            probs = torch.softmax(logits, dim=-1)

            # idx_next: (batch, 1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            # Append the sampled index (idx_next) to the
            # running sequence.
            idx = torch.cat((idx, idx_next), dim=1)

        return self._tokenizer.decode(idx.squeeze(0).tolist())
