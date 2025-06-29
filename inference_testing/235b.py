#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, re, string, torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ──────────── metrics ────────────
def _norm(txt: str) -> str:
    txt = re.sub(r'\b(a|an|the)\b', ' ', txt.lower())
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    return " ".join(txt.split())

def em(pred, gold): return int(_norm(pred) == _norm(gold))

def f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    n = sum((Counter(p) & Counter(g)).values())
    if n == 0: return 0.0
    return 2 * n / (len(p) + len(g))


# ──────────── model ────────────
MODEL = "Qwen/Qwen3-235B-A22B"
tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype="auto", device_map="auto"
)


# ──────────── data ────────────
FILE_PATH = "./dev.parquet"          # 本地 parquet
ds = load_dataset("parquet",
                  data_files={"dev": FILE_PATH},
                  split="dev")

N = 50                            # 调试数量；改成 len(ds) 可全量


# ──────────── inference & scoring ────────────
em_sum = f1_sum = 0
for i, ex in enumerate(tqdm(ds.select(range(N)), desc="evaluating"), 1):
    # 1) 解析 context
    ctx_list = json.loads(ex["context"])           # [[title, [sent1, sent2,...]], ...]
    context  = " ".join(" ".join(p[1]) for p in ctx_list)

    # 2) 组 prompt
    prompt = (
        "Answer the following question using the given context with exactly one phrase. \n\n"
        f"Context:\n{context}\n\nQuestion: {ex['question']}"
    )
    chat = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok([chat], return_tensors="pt").to(model.device)

    # 3) 生成
    with torch.no_grad():
        out = model.generate(**inputs,
                             max_new_tokens=512,
                             do_sample=False)

    gen_ids = out[0][len(inputs.input_ids[0]):]
    try:
        cut = len(gen_ids) - gen_ids[::-1].index(151668)      # 去掉 </think>
    except ValueError:
        cut = 0
    pred = tok.decode(gen_ids[cut:], skip_special_tokens=True).strip()
    print(pred)
    # 4) 评分
    gold = ex["answer"]
    em_val  = em(pred, gold)
    f1_val  = f1(pred, gold)
    em_sum += em_val
    f1_sum += f1_val

    # 5) 打印详情（context 只截前 400 字符，避免太长）
    print("\n" + "-"*80)
    print(f"[Sample {i}]")
    print("Question :", ex["question"])
    print("Gold Ans :", gold)
    print("Pred Ans :", pred)
    # print("Context  :", context[:400] + ("..." if len(context) > 400 else ""))
    print(f"EM={em_val:.0f}, F1={f1_val:.2f}")

print(f"\n── {N} samples ──")
print(f"Exact Match: {em_sum / N:.2%}")
print(f"F1 Score:    {f1_sum / N:.2%}")
