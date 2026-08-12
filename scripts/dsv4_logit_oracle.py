# deepseek_v4 first-token logit oracle — validation target for the Swift port.
import json, mlx.core as mx
from mlx_lm import load
D="/Volumes/models/odysseus/inferencerlabs/DeepSeek-V4-Flash-0731-MLX"
model, tok = load(D)
prompts=["The capital of France is", "def add(a, b):"]
out={}
for p in prompts:
    ids=tok.encode(p)
    import mlx.core as mx
    x=mx.array([ids])
    logits=model(x)[0, -1]              # first-token logits (short prompt = dense regime, exact)
    top=mx.argpartition(-logits, 20)[:20]
    top=sorted([(int(i), float(logits[int(i)])) for i in top.tolist()], key=lambda t:-t[1])
    out[p]={"input_ids":ids, "top20":top}
    print(p, "->", "next_id", top[0][0], tok.decode([top[0][0]]), "logit", round(top[0][1],4))
json.dump(out, open("/tmp/dsv4_oracle.json","w"))
print("ORACLE_SAVED /tmp/dsv4_oracle.json")
