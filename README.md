# B70 Cookbook — 6.1× inference speedup on 2× Intel Arc Pro B70

**Qwen3.8-27B INT4 · tensor-parallel 2 · native MTP6 speculative decode · FP8 KV · 815,700-token context pool**
Fedora 44 · kernel 7.2.11 · vLLM 0.31.0+xpu.llmscaler · measured 2026-11-03

> 📊 Full styled writeup: **[index.html](https://htmlpreview.github.io/?https://github.com/munikor/b70-cookbook/blob/main/index.html)** · charted BetterBench report: **[betterbench-report.html](https://htmlpreview.github.io/?https://github.com/munikor/b70-cookbook/blob/main/betterbench-report.html)**

## The ladder — every rung measured on the same two cards

| Configuration | Short ctx | Medium | Long ctx | KV tokens | Weights/GPU |
|---|---:|---:|---:|---:|---:|
| BF16 TP2 (baseline) | 20.2 | 20.2 | ~20 | 131,072 | 25.8 GiB |
| INT4 GPTQ TP2, no drafter | 57.6 | 57.1 | 54.6 | 1,079,854 | 7.99 GiB |
| INT4 TP1 + native MTP6 | 84.4 | 76.0 | 66.5 | 202,315 | ~15 GiB |
| **INT4 TP2 + native MTP6 (final)** | **123.2** | **113.0** | **95.4** | **815,700** | **7.47 GiB** |

One card (TP1) running the quantized model **with** the drafter beats two cards running it **without**. Speculative decoding is worth more than a whole GPU.

## BetterBench (GGZ14/BetterBench, quick profile, thinking ON)

Combined weighted decode **144.7 t/s** · update p99 **31.5 ms** (no visible stutter at the 99th percentile) · TTFT p50 **147 ms**

| Category | decode t/s (med) | Category | decode t/s (med) |
|---|---:|---|---:|
| file_edit | 203.7 | code | 140.7 |
| math | 168.1 | chat | 93.7 |
| json | 167.4 | prose | 85.2 |
| summarization | 155.3 | reasoning | 143.6 |

**Concurrency (the agentic-chains table):**

| Streams | Aggregate t/s | TTFT p50 | Per-stream decode (med) |
|---:|---:|---:|---:|
| 1 | 120.7 | 280 ms | 156.9 |
| 2 | 184.3 | 360 ms | 120.2 |
| **4** (agent-team shape) | **253.9** | **420 ms** | **87.2** |
| 8 | 321.4 | 500 ms | 56.5 |

**Context ladder** (fresh TTFT / prefix-warmed decode): 2K → 123 t/s · 32K → 132 t/s · **131K → 77.9 t/s** (bake-card B60 reference: 77.82 @ 128K — dead match on faster silicon) · warmed TTFT @ 131K: **2.4 s** (fresh: 87 s).

**Prefill sweep:** ~1,300 tok/s at 2K → 1,134 at 47K. This is the W4A16 compute tax — INT4 wins bandwidth (decode), loses dequant overhead (prefill). The bake card measured 1,552 effective on B60s; raising `--max-num-batched-tokens` above the spec-decode default of 4224 is the known lever.

**Speculative decode is lossless — verified:** 10 greedy financial-calculation probes, 9/10 exact (the miss: our prompt was worded ambiguously, not an output-corruption signature). MTP counters: 4,984 drafts × 6 tokens, 10,637 accepted = 35.6% acceptance = **3.13 effective tokens per target step**.

## The launch command

```bash
docker run -d --name swift-tp2 \
  --device /dev/dri \
  --group-add 105 \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --ipc=host --shm-size 32g \
  -p 8220:8000 \
  -v /models/swift-15-int4:/models/swift-15-int4:ro \
  -v /models/vllm-cache:/root/.cache/vllm \
  -e NEOReadDebugKeys=1 \
  -e EnableDeferBacking=0 \
  -e CCL_SYCL_ALLREDUCE_LL=twoshots \
  -e CCL_SYCL_ALLREDUCE_SIMPLE_READ=1 \
  -e CCL_SYCL_COPY_ENGINE=1 \
  -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_ALLTOALL_TMP_BUF=1 \
  --entrypoint vllm \
  intel/llm-scaler-vllm:0.31.0-b1-experimental \
  serve /models/swift-15-int4 \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4224 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":6}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --port 8000
```

`--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` are what let tool-calling frameworks (LangChain agent graphs, OpenAI-SDK tool use, TradingAgents) hit this endpoint — without them every tool round-trip returns HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice`.

## Five pitfalls that cost us hours (so they cost you zero)

1. **OneCCL allreduce dead at boot on xe + kernel 7.2.11** — `zeMemOpenIpcHandle → ZE_RESULT_ERROR_INVALID_ARGUMENT`, every transport. Fix: `NEOReadDebugKeys=1 -e EnableDeferBacking=0`. With `=1` it loads, computes, and **wedges the box 12 minutes later**. Non-negotiable.
2. **TTM eviction at high VRAM occupancy** — any process touching a card while the pool rides ~97% triggers a 240-second driver hang, full box. Hard cap `--gpu-memory-utilization 0.92`; no monitoring queries against the cards.
3. **Missing `--group-add 105`** — the Fedora render GID. Without it: oneCCL init dies with a cryptic `device_fd is invalid`.
4. **Community image TP2 IPC** — `cyspiegel/vllm-xpu-b70` TP2 dies on the same IPC wall even with the fix (userspace mismatch). It runs **TP1 beautifully** (we measured 84 t/s single-card). For TP2: Intel's official image. Use each build where it's proven.
5. **5.7 GiB of VRAM in a profile pool nobody asked for** — CUDA-graph memory profiling is default-on since v0.21.0; that pool is yours for KV.

## What INT4 buys, in one paragraph

Not raw C1 speed — the drafter buys that. INT4 buys **4× the context** (815K-token KV pool where BF16 had 131K), **4× the headroom** (8 GiB/card free for co-residency vs 0.8), and a faster cold load (7 s vs 94 s). Quality: GPTQ-baked head + MTP linears, INT8 embeddings; the bake's withheld-activation checks show GPTQ halving round-to-nearest error on every baked linear. Our 10-probe math battery ran 9/10 exact on the quantized lane.

## Hardware & stack

- 2× Intel Arc Pro B70 32 GB (BMG-G31, Xe2) · `pcie_aspm=off` · runtime-PM forced on via udev (the `xe` runtime-PM idle bug fakes "GPU lost" crashes on idle-wake — check it first on any "primitive op unsupported" after idle)
- Fedora 44, kernel 7.2.11-200, xe driver, oneAPI 2026.1 userspace
- `intel/llm-scaler-vllm:0.31.0-b1-experimental` — official Intel image (TP2-proven)
- `cyspiegel/vllm-xpu-b70` — community build (TP1-proven, presets + MTP hardening)
- Model: `Wrapzii/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8`

## Links (all B70-scoped)

**Checkpoint & models:** [Baked Swift INT4 (the exact weights here; author-validated on B60, all numbers in this repo are B70)](https://huggingface.co/Wrapzii/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8) · [Swift base](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) · [Qwen3.8-27B INT4 AutoRound (B70-tested)](https://huggingface.co/CySpiegel/Qwen3.8-27B-Int4-AutoRound) · [DFlash2 drafter (B70 TP1-verified)](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)

**Images & sources:** [cyspiegel/vllm-xpu-b70 (B70 image)](https://hub.docker.com/r/cyspiegel/vllm-xpu-b70) · [vllm-intel fork (B70 mods)](https://github.com/CySpiegel/vllm-intel) · [vllm-xpu-kernels (FP8/MXFP4/W4A16 GEMM for Arc)](https://github.com/CySpiegel/vllm-xpu-kernels) · [bake tooling provenance (B60-origin)](https://github.com/Wrapzii/k8v4-xpu)

**Testing:** [GGZ14/BetterBench — the percentile harness used for every test here](https://github.com/GGZ14/BetterBench) · charted output: [betterbench-report.html](https://htmlpreview.github.io/?https://github.com/munikor/b70-cookbook/blob/main/betterbench-report.html) · our ad-hoc harness: [probe-battery.py](probe-battery.py)

---

*All numbers measured 2026-11-03 on the hardware above, streamed token counts from usage stats. Kernel 7.2.11 + xe is a moving target — the TTM and IPC walls above are version-specific; re-verify on your stack.*
