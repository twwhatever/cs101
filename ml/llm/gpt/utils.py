import os
import urllib.request

import torch

from safetensors.torch import load_file
from gpt.mygpt import GPTModel

CACHE_DIR = ".cache/"

URL_DIR = {
  "gpt2-small (124M)": "gpt2",         # works ok
  "gpt2-medium (355M)": "gpt2-medium", # this file seems to have issues via `generate`
  "gpt2-large (774M)": "gpt2-large",   # works ok
  "gpt2-xl (1558M)": "gpt2-xl"         # works ok
}

def _assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.detach())

def _load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = _assign(gpt.pos_emb.weight, params["wpe.weight"])
    gpt.tok_emb.weight = _assign(gpt.tok_emb.weight, params["wte.weight"])

    for b in range(len(gpt.trf_blocks)):
        q_w, k_w, v_w = torch.chunk(
            params[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].attention_block.att.W_q.weight = _assign(
            gpt.trf_blocks[b].attention_block.att.W_q.weight, q_w.T)
        gpt.trf_blocks[b].attention_block.att.W_k.weight = _assign(
            gpt.trf_blocks[b].attention_block.att.W_k.weight, k_w.T)
        gpt.trf_blocks[b].attention_block.att.W_v.weight = _assign(
            gpt.trf_blocks[b].attention_block.att.W_v.weight, v_w.T)

        q_b, k_b, v_b = torch.chunk(
            params[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].attention_block.att.W_q.bias = _assign(
            gpt.trf_blocks[b].attention_block.att.W_q.bias, q_b)
        gpt.trf_blocks[b].attention_block.att.W_k.bias = _assign(
            gpt.trf_blocks[b].attention_block.att.W_k.bias, k_b)
        gpt.trf_blocks[b].attention_block.att.W_v.bias = _assign(
            gpt.trf_blocks[b].attention_block.att.W_v.bias, v_b)

        gpt.trf_blocks[b].attention_block.att.out_proj.weight = _assign(
            gpt.trf_blocks[b].attention_block.att.out_proj.weight,
            params[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].attention_block.att.out_proj.bias = _assign(
            gpt.trf_blocks[b].attention_block.att.out_proj.bias,
            params[f"h.{b}.attn.c_proj.bias"])

        gpt.trf_blocks[b].ff_block.ff.layers[0].weight = _assign(
            gpt.trf_blocks[b].ff_block.ff.layers[0].weight,
            params[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff_block.ff.layers[0].bias = _assign(
            gpt.trf_blocks[b].ff_block.ff.layers[0].bias,
            params[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff_block.ff.layers[2].weight = _assign(
            gpt.trf_blocks[b].ff_block.ff.layers[2].weight,
            params[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff_block.ff.layers[2].bias = _assign(
            gpt.trf_blocks[b].ff_block.ff.layers[2].bias,
            params[f"h.{b}.mlp.c_proj.bias"])

        gpt.trf_blocks[b].attention_block.norm1.weight = _assign(
            gpt.trf_blocks[b].attention_block.norm1.weight,
            params[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].attention_block.norm1.bias = _assign(
            gpt.trf_blocks[b].attention_block.norm1.bias,
            params[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].ff_block.norm2.weight = _assign(
            gpt.trf_blocks[b].ff_block.norm2.weight,
            params[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].ff_block.norm2.bias = _assign(
            gpt.trf_blocks[b].ff_block.norm2.bias,
            params[f"h.{b}.ln_2.bias"])

    gpt.final_norm.weight = _assign(gpt.final_norm.weight, params["ln_f.weight"])
    gpt.final_norm.bias = _assign(gpt.final_norm.bias, params["ln_f.bias"])
    gpt.out_head.weight = _assign(gpt.out_head.weight, params["wte.weight"])


def gpt_from_pretrained_url(config, model):
    url = (
        "https://huggingface.co/openai-community/"
        f"{URL_DIR[model]}/resolve/main/model.safetensors"
    )
    model_file = f"{CACHE_DIR}model-{URL_DIR[model]}.safetensors"
    if not os.path.exists(model_file):
        os.makedirs(CACHE_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, model_file)
    state_dict = load_file(model_file)
    gpt = GPTModel(config)
    _load_weights_into_gpt(gpt, state_dict)
    return gpt


def count_trainable_parameters(model):
  return sum(
      p.numel() for p in model.parameters() if p.requires_grad
  )
