# Building the QAMPARI Corpus and Embeddings

The current QAMPARI setup is wrong in three ways:
1. **Wrong corpus** — uses proof snippets from HF training data (1.32M passages, avg 29 words) instead of the official chunked Wikipedia corpus (~100 words avg)
2. **Wrong pooling** — mean pooling instead of last-token pooling for Qwen3-Embedding-0.6B
3. **Wrong split** — eval questions from training split (capped at 20) instead of validation split (1,000)

This guide rebuilds everything from scratch to match the KARL paper exactly.

## Target (from KARL paper Table 2 + Section 2.2)

| Stat | Value |
|---|---|
| Questions (#Q) | 1,000 (validation split) |
| Avg question tokens | 12.3 |
| Relevant chunks/Q | 14.8 ± 22.9 |
| Answer nuggets/Q | 14.7 ± 23.0 |
| **Indexed document chunks (#D)** | **256,680** |
| **Avg document token length** | **129.8** |
| Embedding model | Qwen3-0.6B |
| Retrieval k | 20 |
| Pooling | last-token |

Paper quote: "For QAMPARI, we use the provided sentence-level chunks (approximately 100 words on average) and index documents containing at least one gold answer entity, resulting in over 250k indexed chunks."

## Data Sources

1. **Chunked Wikipedia** (the corpus): `https://aggreg-qa.s3.amazonaws.com/chunked_wikipedia.tar.gz` (6.4 GB)
   - Format: JSONL files in `wikipedia_chunks/chunks_v5/wikipedia_chunks_*.jsonl`
   - Each line: `{"id": "17177507__0", "contents": "...", "meta": {"url": "...", "title": "..."}}`
   - Contains millions of ~100-word sentence-level chunks

2. **QAMPARI questions** (for eval + entity filtering): `momo4382/QAMPARI` on HuggingFace
   - `validation` split: 1,000 questions (what the paper evaluates on)
   - Each question has `answer_list` containing gold entities

## Prerequisites

```bash
export SHADEFORM_API_KEY="$(python3 -c 'import json; print(json.load(open("~/.konash/config.json".replace("~",__import__("os").path.expanduser("~"))))["shadeform_api_key"])')"
SSH_KEY="~/.ssh/id_ed25519"
SSH_KEY_ID="66f14b43-e883-4ad4-b230-91199bc16429"
```

## Step 1: Launch GPU

A6000 is sufficient. The 0.6B embedding model fits easily in 48GB VRAM.

```bash
INSTANCE_ID=$(curl -s -X POST "https://api.shadeform.ai/v1/instances/create" \
  -H "X-API-KEY: $SHADEFORM_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"cloud\": \"imwt\",
    \"region\": \"desmoines-usa-2\",
    \"shade_instance_type\": \"A6000\",
    \"shade_cloud\": true,
    \"name\": \"qampari-embed\",
    \"ssh_key_id\": \"$SSH_KEY_ID\"
  }" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "Instance: $INSTANCE_ID"
```

Wait ~2 min:

```bash
IP=$(curl -s -H "X-API-KEY: $SHADEFORM_API_KEY" \
  "https://api.shadeform.ai/v1/instances/$INSTANCE_ID/info" | \
  python3 -c "import json,sys; print(json.load(sys.stdin).get('ip',''))")
echo "IP: $IP"
SSH="ssh -i $SSH_KEY shadeform@$IP"
```

## Step 2: Install Dependencies

```bash
$SSH "pip install torch transformers numpy datasets --quiet 2>&1 | tail -1 && echo done"
```

## Step 3: Download Chunked Wikipedia + Extract Gold Entity Chunks

This is the critical step. We download the official chunked Wikipedia corpus, load the validation split entities, and filter to only chunks containing gold entities.

```bash
$SSH << 'REMOTE'
python3 -u << 'PYEOF'
import json, os, tarfile, glob
from datasets import load_dataset

###############################################################################
# Part A: Download and extract chunked Wikipedia
###############################################################################
print("Downloading chunked Wikipedia (6.4 GB)...")
import urllib.request
url = "https://aggreg-qa.s3.amazonaws.com/chunked_wikipedia.tar.gz"
urllib.request.urlretrieve(url, "chunked_wikipedia.tar.gz")
print("Download complete. Extracting...")

os.makedirs("wiki_chunks_raw", exist_ok=True)
with tarfile.open("chunked_wikipedia.tar.gz", "r:gz") as tar:
    tar.extractall("wiki_chunks_raw")
print("Extraction complete.")

# Find all JSONL files
jsonl_files = sorted(glob.glob("wiki_chunks_raw/**/*.jsonl", recursive=True))
print(f"Found {len(jsonl_files)} JSONL files")

###############################################################################
# Part B: Load validation split gold entities
###############################################################################
print("\nLoading QAMPARI validation split...")
ds = load_dataset("momo4382/QAMPARI", split="validation")
print(f"Validation questions: {len(ds)}")

# Collect all gold answer entities (lowercased for matching)
gold_entities = set()
for rec in ds:
    for answer in (rec.get("answer_list") or []):
        if isinstance(answer, dict):
            text = answer.get("answer_text", "").strip()
            if text:
                gold_entities.add(text.lower())

print(f"Unique gold entities: {len(gold_entities):,}")

# Also save eval questions
eval_questions = []
for rec in ds:
    question = rec.get("question_text", "") or rec.get("question", "")
    answer_texts = []
    for answer in (rec.get("answer_list") or []):
        if isinstance(answer, dict):
            answer_texts.append(answer.get("answer_text", ""))
    if question:
        eval_questions.append({"question": question, "answers": answer_texts})

with open("eval_questions.json", "w") as f:
    json.dump(eval_questions, f, indent=2)
print(f"Saved {len(eval_questions)} eval questions")

###############################################################################
# Part C: Filter chunks to those containing at least one gold entity
###############################################################################
print("\nFiltering chunks to gold-entity documents...")
os.makedirs("filtered_chunks", exist_ok=True)

total_chunks = 0
kept_chunks = 0
doc_ids = []
texts = []

for jsonl_file in jsonl_files:
    with open(jsonl_file) as f:
        for line in f:
            total_chunks += 1
            chunk = json.loads(line)
            content = chunk.get("contents", "")
            title = chunk.get("meta", {}).get("title", "")
            chunk_id = chunk.get("id", f"chunk_{total_chunks}")

            # Check if any gold entity appears in the chunk content or title
            content_lower = content.lower()
            title_lower = title.lower()
            found = False
            for entity in gold_entities:
                if entity in content_lower or entity in title_lower:
                    found = True
                    break

            if found and len(content.strip()) >= 5:
                kept_chunks += 1
                doc_ids.append(chunk_id)
                texts.append(content)

                # Save as text file
                safe_id = chunk_id.replace("/", "_").replace("\\", "_")[:120]
                filepath = os.path.join("filtered_chunks", f"{safe_id}.txt")
                with open(filepath, "w", encoding="utf-8") as out:
                    out.write(content)

            if total_chunks % 500000 == 0:
                print(f"  Scanned {total_chunks:,} chunks, kept {kept_chunks:,} ({kept_chunks/total_chunks*100:.1f}%)")

print(f"\nDone!")
print(f"  Total chunks scanned: {total_chunks:,}")
print(f"  Chunks kept (gold entity match): {kept_chunks:,}")
print(f"  Paper target: 256,680")

# Save doc_ids and texts for embedding
import pickle
with open("filtered_data.pkl", "wb") as f:
    pickle.dump({"doc_ids": doc_ids, "texts": texts}, f)
print(f"  Saved filtered data for embedding")

# Word count stats
import random
random.seed(42)
sample = random.sample(texts, min(1000, len(texts)))
word_counts = [len(t.split()) for t in sample]
avg_words = sum(word_counts) / len(word_counts)
print(f"  Avg words/chunk: {avg_words:.0f} (paper says ~100)")
PYEOF
REMOTE
```

**Expected output:** ~256K chunks kept, ~100 words average. If the count is significantly off, the entity matching may need tuning (e.g., case-insensitive substring vs exact match, title-only vs content matching).

## Step 4: Embed with Correct Pooling

Upload the embedding script:

```bash
scp -i $SSH_KEY /Users/joeyroth/Desktop/openkona/scripts/embed_financebench_pages.py shadeform@$IP:~/embed.py
```

Run embedding on the filtered chunks:

```bash
$SSH << 'REMOTE'
nohup python3 -u embed.py \
  --pages-dir ./filtered_chunks \
  --output prebuilt_index.npz \
  --batch-size 256 \
  > embed.log 2>&1 &
echo "PID=$!"
REMOTE
```

**IMPORTANT — verify the embed script uses last-token pooling:**

The script (`embed_financebench_pages.py`) must have this in `embed_batch()`:
```python
# Last-token pooling (CORRECT for Qwen3-Embedding-0.6B)
seq_lens = attention_mask.sum(dim=1) - 1
batch_indices = torch.arange(hidden.size(0), device=hidden.device)
embeddings = hidden[batch_indices, seq_lens.long()]
```

NOT this:
```python
# Mean pooling (WRONG)
mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
embeddings = torch.sum(hidden * mask_expanded, dim=1) / sum_mask
```

Also verify the min char filter is set to 5 (not 50):
```python
if len(text) >= 5:  # Skip bare page numbers, keep real content
```

Monitor progress:

```bash
$SSH "tail -5 embed.log"
```

At 256K chunks with batch_size=256 on A6000 (~77 pages/sec): ~55 minutes.

## Step 5: Download and Verify

```bash
# Download index
scp -i $SSH_KEY shadeform@$IP:~/prebuilt_index.npz /tmp/qampari_index.npz

# Download eval questions
scp -i $SSH_KEY shadeform@$IP:~/eval_questions.json /tmp/qampari_eval_questions.json

# Verify
python3 << 'PYEOF'
import numpy as np, json

data = np.load("/tmp/qampari_index.npz", allow_pickle=True)
print(f"=== Index ===")
print(f"Vectors: {data['vectors'].shape[0]:,} (target: 256,680)")
print(f"Dimensions: {data['vectors'].shape[1]}")
print(f"Embed model: {data['embed_model']}")
print(f"Sample IDs: {data['doc_ids'][:3].tolist()}")

with open("/tmp/qampari_eval_questions.json") as f:
    qs = json.load(f)
print(f"\n=== Eval Questions ===")
print(f"Questions: {len(qs)} (target: 1,000)")
total = sum(len(q['answers']) for q in qs)
print(f"Avg entities/Q: {total/len(qs):.1f} (target: 14.7)")

# Pooling verification
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True).eval()

text = "Which countries have won the FIFA World Cup?"
encoded = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    out = model(**encoded)
mask = encoded["attention_mask"]
seq_lens = mask.sum(dim=1) - 1
last_emb = out.last_hidden_state[0, seq_lens[0].long()].unsqueeze(0)
last_emb = torch.nn.functional.normalize(last_emb, p=2, dim=1).numpy().flatten()

stored = data['vectors'][0].astype(np.float32)
stored = stored / max(np.linalg.norm(stored), 1e-9)
cosine = np.dot(stored, last_emb)
print(f"\nPooling check (cosine with last-token query): {cosine:.4f}")
print(f"Verdict: {'CORRECT (last-token)' if cosine > 0.15 else 'WRONG (likely mean)'}")
PYEOF
```

**Checklist — all must pass:**
- [ ] Vectors ~256K (within 10% of 256,680)
- [ ] Dimensions = 1024
- [ ] Embed model = Qwen/Qwen3-Embedding-0.6B
- [ ] Doc IDs look like `17177507__0` (Wikipedia page ID + chunk index)
- [ ] Eval questions = 1,000
- [ ] Avg entities/Q ~14.7
- [ ] Pooling cosine > 0.15 (last-token matches)

## Step 6: Upload to HuggingFace

```bash
python3 << 'PYEOF'
from huggingface_hub import HfApi
api = HfApi(token="$HF_TOKEN")

# Upload index (overwrites old mean-pooled one)
api.upload_file(
    path_or_fileobj="/tmp/qampari_index.npz",
    path_in_repo="qampari/qwen3-0.6b.npz",
    repo_id="konaeq/konash-indexes",
    repo_type="dataset",
)
print("Index uploaded")
PYEOF
```

## Step 7: Install Locally

```bash
# Install index
cp /tmp/qampari_index.npz ~/.konash/corpora/qampari/prebuilt_index.npz

# Install eval questions (replacing the old 20-question file)
cp /tmp/qampari_eval_questions.json ~/.konash/corpora/qampari/eval_questions.json

# Install filtered chunks as the documents directory
# (need to download from server or rebuild locally)
```

**Note:** The filtered chunk text files also need to be available locally at `~/.konash/corpora/qampari/documents/` (or `pages/`) for the Corpus to resolve doc_ids to text at search time. Options:
- Tar and download from the server
- Upload to HuggingFace as `qampari/chunks.tar.gz` (like we did for FinanceBench pages)
- Re-run the filtering locally

## Step 8: Update download_qampari()

The download function in `konash/download.py` needs updating to:
1. Pull eval questions from the **validation** split (not train, not capped at 20)
2. Pull the prebuilt index from HuggingFace (already does this)
3. Pull the filtered chunk text files from HuggingFace (new — needs a tar like FinanceBench)

## Step 9: Teardown

```bash
curl -s -X POST -H "X-API-KEY: $SHADEFORM_API_KEY" \
  "https://api.shadeform.ai/v1/instances/$INSTANCE_ID/delete"

# Verify (may need multiple attempts)
curl -s -H "X-API-KEY: $SHADEFORM_API_KEY" \
  "https://api.shadeform.ai/v1/instances/$INSTANCE_ID/info" | \
  python3 -c "import json,sys; print(json.load(sys.stdin)['status'])"
```

## Step 10: Smoke Test

```bash
PYTHONPATH=/Users/joeyroth/Desktop/openkona \
OPENAI_API_KEY="$OPENAI_API_KEY" \
TOGETHER_API_KEY="$TOGETHER_API_KEY" \
konash eval qampari \
  --provider together \
  --limit 5 \
  --workers 1
```

With correct embeddings:
- Retrieval scores should be >0.4 (currently ~0.2 with wrong corpus + wrong pooling)
- Entity-relevant chunks should rank in top 5 for most queries
- Score should be directionally close to 45.9% (paper's GLM 4.5 Air base)

## Troubleshooting

### Chunk count too low (<200K)
The entity matching is too strict. Try:
- Match on title only (not content)
- Use partial/fuzzy matching
- Check if entities need normalization (e.g., "United States" vs "U.S.")

### Chunk count too high (>300K)
The entity matching is too loose. Try:
- Require entity match in content (not just title)
- Use word-boundary matching instead of substring
- Check for overly common entity names matching unrelated content

### Retrieval scores still low after re-embedding
- Verify pooling with the cosine check in Step 5
- Check that `_align_embed_fn` loads Qwen3-0.6B for queries (not trigram fallback)
- Verify the query instruction prefix is applied (sentence-transformers does this automatically)

### Download fails or times out
The chunked Wikipedia is 6.4GB. On a cloud GPU with fast internet it should take ~2-3 minutes. If it fails:
- Try `wget --retry-connrefused` instead of `urllib.request`
- Download to ephemeral disk if root is too small
