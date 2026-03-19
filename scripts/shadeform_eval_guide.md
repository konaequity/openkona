# Running FinanceBench Eval on Shadeform

Step-by-step guide to running the FinanceBench evaluation on a Shadeform GPU instance with self-hosted GLM 4.5 Air via vLLM.

## Prerequisites

```bash
# API Keys — stored in ~/.konash/config.json
export SHADEFORM_API_KEY="$(python3 -c 'import json; print(json.load(open("~/.konash/config.json".replace("~",__import__("os").path.expanduser("~"))))["shadeform_api_key"])')"
export OPENAI_API_KEY="$OPENAI_API_KEY"  # set in shell env

# SSH
SSH_KEY="~/.ssh/id_ed25519"
SSH_KEY_ID="66f14b43-e883-4ad4-b230-91199bc16429"  # registered in Shadeform as "local-macbook"
SSH_USER="shadeform"
```

HuggingFace (for konaeq/konash-indexes, write access):
```bash
export HF_TOKEN="$HF_TOKEN"  # set in shell env
```

## 1. Launch Instance

2x H100 (160GB VRAM total). Needed for GLM 4.5 Air FP8 (106B MoE) with TP=2.

```bash
INSTANCE_ID=$(curl -s -X POST "https://api.shadeform.ai/v1/instances/create" \
  -H "X-API-KEY: $SHADEFORM_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"cloud\": \"hyperstack\",
    \"region\": \"montreal-canada-2\",
    \"shade_instance_type\": \"H100x2\",
    \"shade_cloud\": true,
    \"name\": \"financebench-eval\",
    \"ssh_key_id\": \"$SSH_KEY_ID\"
  }" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "Instance: $INSTANCE_ID"
```

Wait ~2 min for status `active`, then get the IP:

```bash
IP=$(curl -s -H "X-API-KEY: $SHADEFORM_API_KEY" \
  "https://api.shadeform.ai/v1/instances/$INSTANCE_ID/info" | \
  python3 -c "import json,sys; print(json.load(sys.stdin).get('ip',''))")
echo "IP: $IP"
```

## 2. Setup Instance

```bash
SSH="ssh -i $SSH_KEY $SSH_USER@$IP"
SCP="scp -i $SSH_KEY"

# Upload repo (run from openkona project root)
cd /Users/joeyroth/Desktop/openkona
tar czf /tmp/konash_repo.tar.gz \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='eval_results' --exclude='dist' --exclude='*.egg-info' \
  --exclude='notebooks' --exclude='tools/trace_viewer/data' \
  konash/ scripts/ pyproject.toml tests/

$SCP /tmp/konash_repo.tar.gz $SSH_USER@$IP:~/

# Setup on instance
$SSH << 'REMOTE'
mkdir -p openkona && tar xzf konash_repo.tar.gz -C openkona/ 2>/dev/null

# IMPORTANT: Symlink HF cache to ephemeral disk (root disk is only 97GB, model is ~65GB)
mkdir -p /ephemeral/hf_cache ~/.cache
rm -rf ~/.cache/huggingface 2>/dev/null
ln -sf /ephemeral/hf_cache ~/.cache/huggingface

# Install deps
pip install vllm datasets huggingface_hub rich numpy --quiet
cd openkona && pip install . --quiet
echo "Done"
REMOTE
```

## 3. Start vLLM

```bash
$SSH << 'REMOTE'
export PATH=$PATH:/home/shadeform/.local/bin
nohup vllm serve zai-org/GLM-4.5-Air-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser glm45 \
  --port 8000 \
  > vllm.log 2>&1 &
echo "vLLM starting"
REMOTE
```

Key flags:
- `--tensor-parallel-size 2` — split model across 2 H100s
- `--max-model-len 131072` — 131K context (needed for multi-search trajectories with k=20 results)
- `--tool-call-parser glm45` — GLM 4.5 specific tool call parser (not `hermes`)
- `--enable-auto-tool-choice` — required for tool calling

Wait ~2 min for model download + loading, then verify:

```bash
$SSH "curl -s http://localhost:8000/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"data\"][0][\"id\"])'"
```

Should print: `zai-org/GLM-4.5-Air-FP8`

## 4. Download Corpus

```bash
$SSH << 'REMOTE'
cd openkona
PYTHONPATH=/home/shadeform/openkona python3 -c "
from konash.download import download_financebench
from rich.console import Console
download_financebench(console=Console())
"
REMOTE
```

Should show: `53,905 page files installed`

## 5. Run Eval

### Quick smoke test (2 questions, verbose)

```bash
$SSH << 'REMOTE'
cd openkona
export PYTHONPATH=/home/shadeform/openkona
export OPENAI_API_KEY="$OPENAI_API_KEY"
nohup konash eval financebench \
  --provider vllm \
  --api-base http://localhost:8000/v1 \
  --model zai-org/GLM-4.5-Air-FP8 \
  --workers 1 \
  --parallel 3 \
  --limit 2 \
  > eval.log 2>&1 &
echo "PID=$!"
REMOTE
```

Monitor the smoke test:
```bash
# Tail the log in real time
$SSH "tail -f openkona/eval.log"

# Or check periodically
$SSH "cat openkona/eval.log"
```

### Full 150-question eval

```bash
$SSH << 'REMOTE'
cd openkona
export PYTHONPATH=/home/shadeform/openkona
export OPENAI_API_KEY="$OPENAI_API_KEY"
nohup konash eval financebench \
  --provider vllm \
  --api-base http://localhost:8000/v1 \
  --model zai-org/GLM-4.5-Air-FP8 \
  --workers 4 \
  --parallel 3 \
  > eval_full.log 2>&1 &
echo "PID=$!"
REMOTE
```

### Monitor progress

```bash
# Latest scores
$SSH "grep '/150' openkona/eval_full.log | tail -5"

# Answers vs references
$SSH "grep -E '(Answer:|Ref:)' openkona/eval_full.log | tail -20"

# Check if still running
$SSH "ps aux | grep 'konash eval financebench' | grep -v grep"

# Kill if needed
$SSH "pkill -f 'konash eval financebench'"
```

### Download results (DO THIS BEFORE TEARDOWN)

```bash
# Always download logs and results before deleting the instance
$SCP $SSH_USER@$IP:~/openkona/eval.log ./eval.log
$SCP $SSH_USER@$IP:~/openkona/eval_full.log ./eval_full.log
$SCP $SSH_USER@$IP:~/openkona/eval_results/financebench_eval.json ./
```

## 6. Teardown

IMPORTANT: Always delete the instance when done. 2xH100 is $3.80/hr.

```bash
curl -s -X POST -H "X-API-KEY: $SHADEFORM_API_KEY" \
  "https://api.shadeform.ai/v1/instances/$INSTANCE_ID/delete"

# Verify deletion
curl -s -H "X-API-KEY: $SHADEFORM_API_KEY" \
  "https://api.shadeform.ai/v1/instances/$INSTANCE_ID/info" | \
  python3 -c "import json,sys; print(json.load(sys.stdin)['status'])"
```

If status is still `active`, retry the delete. Shadeform sometimes needs multiple attempts.

## Common Issues

### `No space left on device`
The root disk is only 97GB. The GLM 4.5 Air FP8 model is ~65GB. Always symlink `~/.cache/huggingface` to `/ephemeral` (1.5TB) before starting vLLM.

### `HTTP Error 400: tool choice requires --enable-auto-tool-choice`
vLLM needs both `--enable-auto-tool-choice` and `--tool-call-parser glm45` flags.

### `HTTP Error 400: input tokens exceed context length`
Increase `--max-model-len`. With k=20 search results at ~700 tokens each, context fills fast. Use 131072.

### `CUDA out of memory` for embedding model
The Qwen3-0.6B embedding model for query encoding tries to load on GPU which is full from vLLM. The corpus `_align_embed_fn` handles this by loading on CPU for 0.6B models. If it still fails, the prebuilt index will work but query embeddings may be degraded.

### `ModuleNotFoundError: No module named 'konash'`
Set `PYTHONPATH=/home/shadeform/openkona` or run `pip install .` from the openkona directory.

## Cost

- 2x H100 on Hyperstack: ~$3.80/hr
- Full 150-question eval (single + parallel N=3): ~45-60 min
- Estimated cost per full eval run: ~$3-4

---

## Architecture Decisions & Fixes (March 2026)

Context for future debugging sessions. These are the issues we hit and the fixes applied.

### Corpus: 53K pages, not 168 documents

The original `download_financebench()` only extracted ~168 evidence snippets from the HuggingFace dataset (the pages referenced by the 150 questions). The KARL paper indexes **all 368 PDFs** from the [FinanceBench GitHub repo](https://github.com/patronus-ai/financebench/tree/main/pdfs), producing **53,905 page-level chunks** (vs the paper's 53,399 — the 506 delta is from slightly different empty-page filtering).

**Fix:** Pre-built page-level index and pages tarball uploaded to `konaeq/konash-indexes` on HuggingFace. The `download_financebench()` function now pulls these instead of building a document-level index from evidence snippets.

**Files changed:** `konash/download.py`, `konash/corpus.py`

### Solver prompt: must be a user message with {question} substituted

The KARL paper's Task Solver Prompt (Figure 34) is a **template** with `Question: {question}` as a placeholder. It is meant to be sent as a **single user message** with the question substituted in.

Previously, the raw template (including the literal string `{question}`) was used as the **system prompt**, and the actual question was a separate user message. This meant:
- The model saw `{question}` as a literal string, not the actual question
- The output format instructions (`Exact Answer:`, `Confidence:`) were in the system prompt, far from the actual question
- The model rarely followed the structured output format

**Fix:** `api.py` `_make_agent()` now sets `system_prompt=None`. The `solve()` method calls `_format_solver_prompt(query)` which substitutes the question into the template, and passes the result as the user message via `env.reset(prompt=formatted_query)`.

**Files changed:** `konash/api.py`

### Answer extraction: parse `Exact Answer:` line

The KARL prompt asks the model to output `Exact Answer: {succinct answer}`. Previously, `extract_final_answer()` just returned the raw last assistant message content, including think tags and long explanations. The judge would then compare a multi-paragraph response against a short nugget like `$8.70`.

**Fix:** `extract_final_answer()` now:
1. Searches for `Exact Answer:` in the raw content (including inside `<think>` tags)
2. If found, extracts just that line
3. Strips `<think>...</think>` tags from the result
4. Falls back to full cleaned content if no `Exact Answer:` line exists

**Files changed:** `konash/agent.py`

### Search results must include doc_id

The 53K-page corpus contains pages from ~60 companies. Financial pages across companies use nearly identical language ("Property, Plant and Equipment", "Cash Flow Statement", etc.). When searching for "3M FY2018 balance sheet PP&E", the correct 3M page ranked #11 out of 20 — behind PepsiCo, Pfizer, Nike, Intel, etc.

Most pages don't mention the company name in the text body. Without doc_ids in the search results, the model had no way to distinguish which result belonged to which company.

**Fix:** Search results now include the source filename: `[1] (score: 0.803) [3M_2018_10K_p46.txt] Table of Contents...`. The model can see the company name and filing year in the doc_id and skip irrelevant results.

**Files changed:** `konash/api.py` (tool_executor in `solve()`)

### No trigram fallback for embeddings

The `load_embedding_model()` function previously fell back to a character-trigram hash embedding when the real model couldn't load. This produced vectors in a completely different space from the prebuilt Qwen3-0.6B index, making search results random noise while appearing to work (scores still returned, just meaningless).

**Fix:** Removed the trigram fallback. `load_embedding_model()` now raises `RuntimeError` if the model can't load. The `_make_embed_fn()` in `api.py` catches this and returns `None`, letting the prebuilt index's `_align_embed_fn` handle query embedding.

**Files changed:** `konash/retrieval/vector_search.py`, `konash/api.py`

### vLLM configuration for GLM 4.5 Air

Critical vLLM flags that must be set correctly:
- `--tool-call-parser glm45` — NOT `hermes`. vLLM has a GLM-4.5-specific parser.
- `--max-model-len 131072` — 32K is too small. With k=20 search results at ~700 tokens each, context overflows after 2-3 searches.
- `--enable-auto-tool-choice` — required for any tool calling.
- HF cache **must** be on ephemeral disk — root disk is 97GB, model is ~65GB.

### KARL paper baseline target

GLM 4.5 Air (base, no RL training) on FinanceBench: **72.7%** (Table 4). This is the number to target. FinanceBench is an **out-of-distribution** task — KARL is never trained on financial data.

### What we haven't verified yet

- Whether the compression plugin threshold (150K chars) is appropriate for FinanceBench (it was tuned for BrowseComp-Plus which has shorter chunks)
- Whether `--workers 4` degrades vLLM response quality under concurrent load
- Whether the `_align_embed_fn` successfully loads Qwen3-0.6B on CPU when vLLM occupies the GPU
- Full 150-question eval results with all fixes applied
