"""功能3 簇标签 —— 本地 Qwen（transformers）给每个聚类起干净唯一的话题名。

c-TF-IDF 关键词标签的硬伤：一堆 LLM 簇都叫 "language / large language"，且 PDF 连字
伪词（effcient/unifed）混进来。改用本地 LLM：喂每簇代表性标题 → 输出 2-5 词英文主题
名，大簇优先、显式避重 → 干净唯一可读。

**在 4090 本机跑**（模型在 /data/arc/.cache/modelscope/Qwen/...）。用空闲 GPU（默认
CUDA_VISIBLE_DEVICES=3 那张空 4090；跑前必 nvidia-smi 确认非他人占用 — W63 共享盒铁律）。

被 build_landscape.py --llm-label 懒加载调用；也可独立测试。
"""
from __future__ import annotations

import os
import re

DEFAULT_MODEL = "/data/arc/.cache/modelscope/Qwen/Qwen3-4B-Instruct-2507"

SYSTEM = (
    "You label clusters of computer-science research papers. Given a sample of "
    "paper titles from ONE cluster, reply with a SHORT topic label (2 to 5 words) "
    "naming the shared research theme. Use lowercase, no punctuation, no quotes, "
    "no trailing period. Reply with ONLY the label."
)


def load_llm(model_path: str = DEFAULT_MODEL, device: str = "cuda"):
    """加载 Qwen（bf16）。返回 (model, tokenizer)。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, tok


def _clean(text: str) -> str:
    text = text.strip().strip('"\'`').strip()
    text = text.splitlines()[0] if text else text
    text = re.sub(r"[.。]+$", "", text).strip()
    return text.lower()[:48]


def label_one(model, tok, titles, *, avoid=None, max_titles=18) -> str:
    """给一簇标题起一个标签，尽量避开 avoid 里已用过的名字。"""
    import torch

    sample = [t.strip()[:110] for t in titles if t and t.strip()][:max_titles]
    body = "\n".join(f"- {t}" for t in sample)
    user = f"Titles from one cluster:\n{body}\n\nTopic label:"
    if avoid:
        user = (f"Already-used labels (pick something DISTINCT, do not repeat): "
                f"{', '.join(sorted(avoid))}\n\n" + user)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return _clean(gen)


def llm_label_clusters(cluster_titles: dict, *, model_path: str = DEFAULT_MODEL,
                       sizes: dict | None = None) -> dict:
    """{cid: [titles]} -> {cid: {"label": str}}。大簇优先起名，后簇显式避重。"""
    model, tok = load_llm(model_path)
    order = sorted(cluster_titles, key=lambda c: (sizes or {}).get(c, 0), reverse=True)
    used: set[str] = set()
    out: dict = {}
    for c in order:
        lab = label_one(model, tok, cluster_titles[c], avoid=used) or f"cluster {c}"
        # 撞名兜底：再问一次带更强避重，仍撞就加序号
        if lab in used:
            lab2 = label_one(model, tok, cluster_titles[c], avoid=used | {lab})
            lab = lab2 if lab2 and lab2 not in used else f"{lab} {c}"
        used.add(lab)
        out[c] = {"label": lab}
        print(f"  [c{c}] {lab}", flush=True)
    return out


if __name__ == "__main__":
    import json
    import sys

    # 独立测试：python llm_label.py cluster_titles.json -> 打印标签
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    ct = {int(k): v for k, v in data["cluster_titles"].items()}
    sz = {int(k): v for k, v in data.get("sizes", {}).items()}
    res = llm_label_clusters(ct, sizes=sz)
    print(json.dumps({str(k): v for k, v in res.items()}, ensure_ascii=False, indent=2))
