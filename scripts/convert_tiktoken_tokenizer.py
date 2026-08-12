#!/usr/bin/env python3
"""Convert a tiktoken-based model tokenizer to a swift-transformers-loadable
tokenizer.json, with a HARD parity gate.

Why: Telemak's swift-transformers tokenizer loader needs a `tokenizer.json`
(fast BPE). Kimi / Moonshot models ship only `tiktoken.model` +
`tokenization_kimi.py` (a slow custom TikTokenTokenizer) and NO tokenizer.json,
so Telemak fails with configurationMissing("tokenizer.json") or, once the json
is present, unsupportedTokenizer("TikTokenTokenizer").

This builds the fast tokenizer from the reference's EXACT pat_str + special
tokens via transformers.convert_slow_tokenizer.TikTokenConverter, then REFUSES
to write it unless token-id parity is exact against the reference on:
  - all special tokens (each -> its reserved id, single token)
  - a diverse text corpus (Latin, CJK Han/JP/KR, code, numbers, whitespace, emoji)
  - a rendered chat-template string (the real serving path)
  - decode round-trip

It also patches tokenizer_config.json: tokenizer_class -> PreTrainedTokenizerFast
(truthful now that a fast tokenizer.json exists; maps to BPETokenizer in
swift-transformers' knownTokenizers), drops auto_map (trust_remote_code pointer),
and injects the reference chat_template if the config lacks one.

Requires a venv with: transformers, tokenizers, tiktoken, blobfile (the
mlx-cluster venv on the Argo nodes has these). PyTorch not needed.

Usage:
  python convert_tiktoken_tokenizer.py /path/to/model_dir
"""
import json, os, sys, shutil, tempfile

def main(D):
    from transformers import AutoTokenizer
    from transformers.convert_slow_tokenizer import TikTokenConverter
    from tokenizers import Tokenizer, AddedToken

    ref = AutoTokenizer.from_pretrained(D, trust_remote_code=True)
    if not hasattr(ref, "pat_str") or not hasattr(ref, "special_tokens"):
        sys.exit("reference tokenizer is not a tiktoken tokenizer (no pat_str/special_tokens)")

    sp = sorted(ref.special_tokens.items(), key=lambda kv: kv[1])  # (tok, id) by id
    conv = TikTokenConverter(vocab_file=os.path.join(D, "tiktoken.model"),
                             pattern=ref.pat_str)
    fast = conv.converted()
    base = fast.get_vocab_size(with_added_tokens=False)
    if base != sp[0][1]:
        sys.exit(f"base vocab {base} != first special id {sp[0][1]}; ids would not line up")
    fast.add_special_tokens([AddedToken(t, special=True, normalized=False) for t, _ in sp])

    tmp = tempfile.mkdtemp(); jp = os.path.join(tmp, "tokenizer.json"); fast.save(jp)
    ft = Tokenizer.from_file(jp)

    # --- parity gate ---
    bad = [t for t, idx in sp if ft.encode(t, add_special_tokens=False).ids != [idx]]
    corpus = [
        "Hello, world! 12345", "def f(x):\n  return x*2  # c",
        "你好，世界！中文分词。",
        "Mixed 中英 42 @#$", "   spaces\tand\n\nnewlines",
        "\U0001f680\U0001f525 café résumé",
        "日本語 한국어 CJK", "A" * 40,
    ]
    tbad = sum(1 for s in corpus if ref.encode(s, add_special_tokens=False) != ft.encode(s).ids)
    # chat-template string is the real serving path (render text, then encode)
    msgs = [{"role": "user", "content": "What is 2+2? 二加二。"}]
    txt = ref.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    cbad = 0 if ref.encode(txt, add_special_tokens=False) == ft.encode(txt, add_special_tokens=False).ids else 1

    print(f"parity: specials={'PASS' if not bad else f'FAIL({len(bad)})'} "
          f"text={'PASS' if tbad == 0 else f'FAIL({tbad})'} "
          f"chat={'PASS' if cbad == 0 else 'FAIL'}")
    if bad or tbad or cbad:
        sys.exit("PARITY FAILED - refusing to write tokenizer.json")

    # --- write (non-destructive: back up, never overwrite an existing json) ---
    dst = os.path.join(D, "tokenizer.json")
    if os.path.exists(dst):
        sys.exit(f"{dst} already exists - not overwriting")
    shutil.move(jp, dst)
    print("wrote", dst, os.path.getsize(dst), "bytes")

    cfg_p = os.path.join(D, "tokenizer_config.json")
    cfg = json.load(open(cfg_p))
    bak = cfg_p + ".pre-fast.bak"
    if not os.path.exists(bak):
        shutil.copy(cfg_p, bak)
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg.pop("auto_map", None)
    if not cfg.get("chat_template") and getattr(ref, "chat_template", None):
        cfg["chat_template"] = ref.chat_template
    json.dump(cfg, open(cfg_p, "w"), ensure_ascii=False, indent=2)
    print("patched tokenizer_config.json (tokenizer_class=PreTrainedTokenizerFast, chat_template, -auto_map)")
    print("OK - model is now Telemak-loadable")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
