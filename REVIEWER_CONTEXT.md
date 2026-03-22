# Reviewer Context — Parameter Golf Submission (Round 2)

## Answers Incorporated from Round 1

| # | Idea | Reviewer Verdict | Status |
|---|---|---|---|
| 1 | Learnable quant clipping | YES — per-tensor γ on rowwise scales | **Implemented, smoke testing** |
| 2 | BigramHash broken? | NO — XOR-prime hash is fine | No change needed |
| 3 | Byte-normalized loss | WRONG — per-token byte weighting is incorrect bpb gradient | **Dropped** |
| 4 | Exact bigram logit head | YES — with gate, boundary mask, count init | Implemented but over budget |
| 5 | Sensitivity-based bit allocation | YES — use true compressed bytes | Not yet implemented |

---

## What We Built Since Round 1

### 1. QAT with Learnable Clip Multipliers (NEW)

Per your recommendation: learn a per-tensor γ multiplier on rowwise amax, constrained [0.5, 1.5], regularized toward 1.0.

**Hyperparameters:**
```python
qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "1")))
qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.75))  # start at 75% of training
qat_gamma_lr = float(os.environ.get("QAT_GAMMA_LR", 0.01))
qat_gamma_reg = float(os.environ.get("QAT_GAMMA_REG", 0.01))    # regularize log(γ)²
```

**Core fake-quantize with STE:**
```python
def _fake_quantize_ste(w: Tensor, clip_range: int, gamma: Tensor) -> Tensor:
    row_amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = (gamma * row_amax) / clip_range
    q = torch.round(w / scale).clamp(-(clip_range + 1), clip_range)
    # STE: use quantized value in forward, gradient passes through w
    return w + (q * scale - w).detach()
```

**Apply/restore around forward pass:**
```python
def _apply_qat(base_model, clip_gammas, alpha):
    saved = []
    with torch.no_grad():
        for name, p in base_model.named_parameters():
            if p.ndim == 2 and p.numel() > 65_536 and name in clip_gammas:
                orig = p.data.clone()
                clip_range = 15 if ".mlp." in name else 31  # int5 MLP, int6 attn
                w_q = _fake_quantize_ste(p.data, clip_range, clip_gammas[name])
                p.data.lerp_(w_q, alpha)  # gradual ramp-in
                saved.append((p, orig))
    return saved

def _restore_qat(saved):
    with torch.no_grad():
        for p, orig in saved:
            p.data.copy_(orig)
```

**Gamma creation (after model init):**
```python
clip_gammas: dict[str, nn.Parameter] = {}
for name, p in base_model.named_parameters():
    if p.ndim == 2 and p.numel() > 65_536:
        # Log-space: γ = exp(param), init param=0 → γ=1.0
        gamma = nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32))
        clip_gammas[name] = gamma

optimizer_gamma = torch.optim.Adam(
    [{"params": list(clip_gammas.values()), "lr": 0.01, "base_lr": 0.01}],
    betas=(0.9, 0.999),
)
```

**In training loop:**
```python
train_frac = step / max(args.iterations, 1)
qat_active = (args.qat_enabled and clip_gammas and train_frac >= args.qat_start_frac)
qat_alpha = min((train_frac - args.qat_start_frac) / 0.05, 1.0) if qat_active else 0.0

# Convert log-space to positive, clamp [0.5, 1.5]
effective_gammas = {name: torch.clamp(torch.exp(g), 0.5, 1.5)
                    for name, g in clip_gammas.items()} if qat_active else {}
qat_saved = _apply_qat(base_model, effective_gammas, qat_alpha) if qat_active else []

# ... forward + backward ...

# Gamma regularization: keep γ near 1.0
if qat_active and micro_step == grad_accum_steps - 1:
    gamma_reg = sum(g.square().sum() for g in clip_gammas.values())
    loss = loss + 0.01 * gamma_reg

# Restore original weights after backward
if qat_saved:
    _restore_qat(qat_saved)
```

**In final quantization (use learned γ):**
```python
final_gammas = {}
for name, g in clip_gammas.items():
    final_gammas[name] = float(torch.clamp(torch.exp(g), 0.5, 1.5).item())

# quantize_intN_per_row now accepts gamma:
def quantize_intN_per_row(t, clip_range=31, gamma=1.0):
    row_max = t.abs().amax(dim=1)
    scale = (gamma * row_max / clip_range).clamp_min(1e-12).to(torch.float16)
    q = torch.clamp(torch.round(t / scale.float()[:, None]), -(clip_range+1), clip_range).to(torch.int8)
    return q, scale

# mixed_quantize_int6 passes learned gamma per tensor:
gamma = learned_gammas.get(name, 1.0) if learned_gammas else 1.0
q, s = quantize_intN_per_row(t, clip_range=clip, gamma=gamma)
```

**Concern:** The `_apply_qat` does fake-quant inside `torch.no_grad()` and modifies `p.data` directly. But `_fake_quantize_ste` uses the STE trick (`w + (q*scale - w).detach()`). Since we're in no_grad and modifying `.data`, the STE gradient doesn't flow. **Is this correct, or should fake-quant happen inside the forward pass (inside autocast) instead of as a weight replacement?**

---

### 2. BigramLogitHead (implemented but disabled)

```python
class BigramLogitHead(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(vocab_size, vocab_size, dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, prev_tokens: Tensor) -> Tensor:
        return self.table[prev_tokens] * self.scale
```

Used in GPT.forward:
```python
logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
if self.bigram_logit is not None:
    prev = input_ids.reshape(-1)
    logits = logits + self.bigram_logit(prev)
```

**Problem:** Adds ~0.78 MB after int6+zstd. We have 0.66 MB headroom. Over by 0.12 MB.

---

### 3. Compression Regularization (kept but likely weak)

```python
def compression_reg(model: nn.Module) -> Tensor:
    penalty = torch.zeros((), device=next(model.parameters()).device)
    count = 0
    for p in model.parameters():
        if p.ndim == 2 and p.numel() > 65536:
            row_range = p.abs().amax(dim=1)
            penalty = penalty + row_range.mean()
            count += 1
    return penalty / max(count, 1)
```

---

### Smoke Test Results

| Run | Features | Artifact | Fits? | Headroom |
|-----|----------|----------|-------|----------|
| v1 | TrigramHash(16384) | 17.05 MB | NO | -1.05 MB |
| v2 | BigramLogitHead | 16.06 MB | NO | -0.12 MB |
| v3 | Baseline (no bigram logit) | 15.28 MB | YES | +0.66 MB |
| v4 | QAT + learnable clip | RUNNING | — | — |

---

## Questions for Round 2

### Q1: STE Gradient Path — Is Our QAT Implementation Correct?

Our `_apply_qat` replaces weight `.data` inside `torch.no_grad()`, then runs the forward pass with modified weights, then restores after backward.

The STE line is:
```python
return w + (q * scale - w).detach()
```

But this runs inside `torch.no_grad()` during `_apply_qat`, so the STE detach doesn't matter — there's no gradient tape. The forward pass then uses the modified `p.data` (which is the quantized version). Gradients flow through the forward pass w.r.t. the quantized weights, but since we restore the original weights afterward, the optimizer step applies to the original (unquantized) weights.

**Is this the correct pattern?** It seems like the gradients are computed on the quantized forward pass but applied to the unquantized weights. The model "sees" quantized weights during forward, but the actual parameters being optimized are the float ones. This is how the upstream `_apply_qat_noise` / `_restore_qat_weights` works too.

But **the γ parameters never get gradients this way** — they're used inside `no_grad()`. So they can't learn. **This is a bug, right?**

If so, the fix would be: don't use the data-replacement trick for γ learning. Instead, integrate fake-quant into the model's forward pass (inside CastedLinear) where the gradient tape is active.

### Q2: Document Boundary Detection

Training data is packed sequences from `TokenStream` — contiguous tokens from pre-tokenized binary shards, no explicit document markers:

```python
class TokenStream:
    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)
```

**How do we detect document boundaries for bigram masking?** There are no BOS/EOS markers in the stream. Options:
- (a) The tokenizer might encode a boundary token — need to check SentencePiece for BOS/EOS IDs
- (b) Heuristic: mask at shard boundaries (but packing crosses shards)
- (c) Just skip boundary masking for now

### Q3: Count-Based Initialization for Bigram Table

You recommended initializing `B[a,b] = log p(b|a) - log p(b)` from corpus counts. With vocab=1024, we need to count 1M bigram pairs.

Training sees ~16M tokens in warmup (20 steps × 786K). That's ~15 observations per bigram on average — quite sparse.

**Alternative:** Count during the first 200 training steps (~157M tokens, ~150 per bigram), then initialize the table and unfreeze it. But the model trains without bigram bias for 200 steps.

**Is 200 steps of counting enough? Or should we precompute counts offline and hard-code the init table?**

### Q4: Sensitivity-Based Bit Allocation — Calibration Budget

Under the 10-minute wallclock, training uses ~590s. That leaves ~10s for calibration.

30 tensors × 3 bit candidates × 2 calibration batches = 180 forward passes. At ~50ms each on 8×H100 ≈ 9s. Tight but feasible.

**Should calibration happen:**
- (a) Within the 10-min wallclock (after training, before final quant)
- (b) Offline as a separate step (find optimal assignment, hard-code it)

Option (b) is simpler — run calibration once, find optimal int4/int5/int6 assignment per tensor, hard-code the result. But it's not adaptive to different runs.

**Which approach?**

### Q5: BigramLogitHead — Fitting Under 16MB

We need ~0.12 MB of space. Options:
1. Sensitivity allocation moves some tensors int6→int5 (frees ~0.15 MB per tensor)
2. Low-rank factorization: `B = U @ V` where U is 1024×r, V is r×1024. At r=64: 131K params (87% smaller)
3. Quantize bigram table to int5 instead of int6 (saves ~20%)
4. Just drop BigramHash (10240×128 = 1.3M params) and replace with BigramLogitHead (1M params) — net savings

**Your recommendation?** Option 4 seems cleanest — the BigramLogitHead directly models transitions and makes BigramHash redundant (BigramHash injects into embeddings, BigramLogitHead injects into logits — they overlap).

### Q6: Compression Regularization — Keep or Remove?

Current `compression_reg()` penalizes `row_range.mean()` (smooth L-infinity). With learnable clip multipliers now handling quantization quality directly, **is this redundant?** Should we remove it to simplify, or does it serve a different purpose (encouraging tighter weight distributions for better zstd)?
