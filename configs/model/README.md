# configs/model

HuggingFace `config.json` files for every model LLMServingSim knows
about. Shared between the **simulator** (for memory-model sizing,
layer counting, MoE routing) and the **profiler** (for picking a
matching architecture yaml and feeding vLLM).

Path convention: `configs/model/<org>/<name>.json` mirrors the HF
repo id. The file that lives at `meta-llama/Llama-3.1-8B.json`
describes `meta-llama/Llama-3.1-8B`.

## What goes in

The full raw `config.json` from the model's HuggingFace repo. vLLM
consumes it directly via a temp directory at profile time, so the
file must contain every field vLLM needs to instantiate the model
under `load_format=dummy`:

| Field | Purpose |
| --- | --- |
| `architectures` | vLLM picks the ForCausalLM class from this list |
| `model_type` | Profiler picks the matching `profiler/models/<model_type>.yaml` |
| `hidden_size`, `intermediate_size` | Linear dims |
| `num_attention_heads`, `num_key_value_heads` | Attention shapes (GQA) |
| `num_hidden_layers` | Layer count (simulator multiplies per-layer time by this) |
| `vocab_size`, `max_position_embeddings` | Embedding + context |
| `head_dim` | Needed when `hidden_size ≠ num_attention_heads × head_dim` (Qwen3) |
| `rms_norm_eps` / `layer_norm_eps` | Norm config |
| `hidden_act` | MLP activation |
| `rope_theta`, `rope_scaling` | Rotary embedding setup (critical for Llama 3's rope_type) |
| `tie_word_embeddings` | Whether lm_head shares weights with embedding |
| `attention_bias`, `mlp_bias` | Linear layer biases |
| `torch_dtype` | Profiler auto-derives variant folder name from this |
| `num_local_experts` or `num_experts`, `num_experts_per_tok`, `moe_intermediate_size` | MoE only |

Leave everything verbatim from the HF repo — the profiler / simulator
ignore keys they don't need, so extra fields are harmless. The only
hard requirements are `architectures`, `model_type`, and the
dimensional fields.

## Currently provided

| File | Type | Layers | Hidden | Heads | KV | MoE |
| --- | --- | --- | --- | --- | --- | --- |
| `meta-llama/Llama-3.1-8B.json` | dense | 32 | 4096 | 32 | 8 | — |
| `meta-llama/Llama-3.1-70B.json` | dense | 80 | 8192 | 64 | 8 | — |
| `Qwen/Qwen3-32B.json` | dense | 64 | 5120 | 64 | 8 | — |
| `Qwen/Qwen3-30B-A3B-Instruct-2507.json` | MoE | 48 | 2048 | 32 | 4 | 128E / top-8 |
| `deepseek-ai/DeepSeek-V3.json` | MoE | 61 | 7168 | 128 | MLA | 256E / top-8 + 1 shared |
| `mistralai/Mixtral-8x7B-v0.1.json` | MoE | 32 | 4096 | 32 | 8 | 8E / top-2 |
| `microsoft/Phi-mini-MoE-instruct.json` | MoE | 32 | 4096 | 32 | 8 | 16E / top-2 |

## Adding a new model

Three ways:

**1. Auto-download (easiest)** — run the profiler with
`MODEL="<org>/<name>"` and `HF_TOKEN` set. If the config isn't
present locally, the profiler fetches it from the HuggingFace hub
and caches it here.

**2. Manual download via Docker** — inside the container:

```bash
python3 -c "
from huggingface_hub import hf_hub_download; import shutil
src = hf_hub_download(repo_id='google/gemma-2-9b', filename='config.json')
shutil.copyfile(src, '/workspace/configs/model/google/gemma-2-9b.json')
"
```

**3. Custom model shape** — hand-write a JSON with the dimensions you
want to profile. Must include `architectures` (for vLLM) and
`model_type` (for the profiler's architecture dispatch). Any of the
existing configs is a working template:

```jsonc
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 16384,
  "intermediate_size": 53248,
  "num_attention_heads": 128,
  "num_hidden_layers": 80,
  "num_key_value_heads": 16,
  "vocab_size": 128256,
  "max_position_embeddings": 32768,
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "hidden_act": "silu"
  // … any other fields vLLM's model class expects
}
```

Save as e.g. `configs/model/custom/my-300b.json`, set
`MODEL="custom/my-300b"` in `profiler/profile.sh`, and
run. The profiler will feed this config to vLLM directly.

## Architecture support

The profiler only runs when a matching architecture yaml exists at
`profiler/models/<model_type>.yaml`. Currently supported
`model_type` values:

* `llama` — Llama 3.x family (uses `Llama3RotaryEmbedding`)
* `qwen3` — Qwen3 dense family
* `qwen3_moe` — Qwen3 MoE family
* `deepseek_v3` — DeepSeek V3 MoE family; current catalog is intended for
  serving/HBFrame front-end integration first. Full profiler use should add
  MLA-specific projection sizing before treating CUDA-profiled latency as final.
* `mixtral` — Mixtral family
* `phimoe` — Phi MoE family

Any other `model_type` (e.g. `gemma2`) produces a clear error at profile
time with instructions for adding support.
