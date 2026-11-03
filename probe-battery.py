import json, time, urllib.request, concurrent.futures as cf

URL = "http://localhost:8220/v1/chat/completions"

def req(prompt, maxtok, stream=True, model="swift-int4"):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": maxtok, "temperature": 0, "stream": stream,
                       "stream_options": {"include_usage": True} if stream else None}).encode()
    return urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})

def stream_bench(prompt, maxtok):
    t0 = time.time(); ttft = None; n = 0
    for line in urllib.request.urlopen(req(prompt, maxtok), timeout=900):
        s = line.decode().strip()
        if not s.startswith("data: "): continue
        d = s[6:]
        if d == "[DONE]": break
        j = json.loads(d)
        if j.get("usage"): n = j["usage"]["completion_tokens"]
        for c in (j.get("choices") or []):
            if c.get("delta", {}).get("content") is not None and ttft is None:
                ttft = time.time() - t0
    el = time.time() - t0
    return n, ttft, el

print("=== 1. DECODE LADDER (length sweep, fresh prefix) ===")
for mt in (256, 1024, 2048, 4096):
    n, ttft, el = stream_bench("Write a detailed, well-structured essay on how clearing houses manage counterparty risk in derivatives markets, with concrete examples.", mt)
    print(f"out={mt:>4}: tokens={n} ttft={ttft:.2f}s rate={(n-1)/(el-ttft):.1f} t/s")

print("=== 2. CONTEXT LADDER (fresh TTFT + warmed decode; card-comparable) ===")
for target in (2048, 8192, 32768, 131072):
    filler = ("The quarterly filing discusses revenue recognition, lease liabilities, goodwill impairment testing, and segment reporting under ASC 606 and 842. " * 45)
    prefix = filler * max(1, target // 680)
    prompt = prefix[:target * 4] + " In one short paragraph, list the three most audit-sensitive areas mentioned."
    n, ttft, el = stream_bench(prompt, 128)
    fresh_ptoks = n  # informational
    n2, ttft2, el2 = stream_bench(prompt, 512)  # warmed (prefix cached)
    rate2 = (n2 - 1) / (el2 - ttft2)
    print(f"ctx~{target:>6}: fresh_ttft={ttft:.2f}s warmed_ttft={ttft2:.2f}s warmed_decode={rate2:.1f} t/s")

print("=== 3. CONCURRENCY (8 parallel, aggregate) ===")
def one(i):
    n, ttft, el = stream_bench(f"Explain in detail why basis trading works in Treasury futures, answer number {i}.", 512)
    return n, el
t0 = time.time()
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, range(8)))
wall = time.time() - t0
tot = sum(r[0] for r in res)
print(f"8x512: total_tokens={tot} wall={wall:.1f}s aggregate={tot/wall:.1f} t/s  per-stream~{tot/8/ (sum(r[1] for r in res)/8):.1f}")

print("=== 4. CORRECTNESS (greedy, spec-decode losslessness) ===")
qs = [
    ("Revenue 400, COGS 260. Gross margin %? One line.", "35"),
    ("Net income 80 on equity 500. ROE %? One line.", "16"),
    ("Position 3% of 1,200,000 account. Size in dollars? One line.", "36,000"),
    ("Entry 142.50, stop 138.20, size 250 shares. Risk dollars? One line.", "1075"),
    ("EV/EBITDA: EV 900, EBITDA 75. Ratio? One line.", "12"),
    ("Free cash flow 45, shares 300. FCF per share? One line.", "0.15"),
    ("Price 118, book 95. P/B two decimals? One line.", "1.24"),
    ("Coupon 4% on 1000 face. Annual interest? One line.", "40"),
    ("Buy 100 at 55, sell 100 at 61. Gross P&L? One line.", "600"),
    ("Sigma 22, daily, sqrt(252) annualization already given 15.87. Daily vol %? One line.", "1.39"),
]
ok = 0
for q, expect in qs:
    r = urllib.request.urlopen(req(q, 128, stream=False), timeout=120)
    d = json.loads(r.read())
    txt = d["choices"][0]["message"]["content"]
    hit = expect.replace(",", "") in txt.replace(",", "").replace(" ", "").replace("%", "")
    ok += hit
    if not hit: print(f"  MISS: {q[:40]} -> {txt[-60:]}")
print(f"correctness: {ok}/10")
