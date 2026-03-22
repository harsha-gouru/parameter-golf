
Considering calibration options

Counting 64 million pairs with torch.bincount could take some time, but it seems manageable. I’m thinking it might be better to initialize from factorized count features rather than a full table. For Q3, option (b) seems safer by using offline predetermined bit assignments, especially in a dev environment. There could be training randomness, so a hybrid approach might work well. I’ll note that the document should highlight deriving default assignments offline while allowing optional online adjustments.Evaluating bigram table options

The artifact’s 10-minute run likely only includes the final training and not offline tuning, which is good. The quality of a low-rank bigram table depends on its rank. The exact bigram matrix might act like a stochastic matrix with low intrinsic rank. I think rank 64 can capture common structures but might miss specifics. Maybe using rank 96 or 128 is safer. I could also consider quantizing the bigram table to int4 or 3-bit instead of overcomplicating it with low-rank techniques.Suggesting improvements

I need to consider a few suggestions. Before factorizing, I should think about dropping BigramHash or reducing its bucket count, and quantizing the bigram table to int4. Tying bigram table rows to embeddings might be more straightforward, making low-rank techniques unnecessary. I should probably remove compression regularization unless it shows size benefits without harming bpb. Disabling row_range penalty by default could simplify things, and learnable clipping may take precedence. Finally, I'll need a concrete recipe for post-quant calibration and temperature scaling to optimize performance.Adjusting quantization methods

For each quantized linear layer, I usually compare the float output and quantized output during calibration by subtracting the mean error from the output channels. Then I can fold that into the following norm or output bias if it’s present. If there's no bias included in the artifact, I might not need it. However, a small per-channel output correction vector could help with sensitive tensors if size allows. I want to prioritize temperature adjustments and ensure clarity around gamma values during final quantization. Plus, I need to check on the BigramLogitHead class being disabled.Refining artifact table suggestions

I could mention that the artifact table's val_bpb at 200 steps is not very relevant for size or quality decisions. It might be better to remove it or at least annotate it as "smoke-test only, not predictive." Keeping it could mislead reviewers since a 200-step bpb of 1.94 isn't useful. For comparisons between the v1 trigram and v2 bigram heads at 200 steps, those aren’t indicative either. It’s worth suggesting a fast pre-pass over 8 to 32 million tokens to accumulate counts before training begins, which should be manageable.Planning a training pre-pass

I could conduct a count pre-pass on the training CPU while reading the same shards. It seems manageable to derive a count matrix with about 1,048,576 entries, using 4 bytes per int32 for a total of 4MB. Using bigram and unigram counts from the first 8 to 32 million tokens with additive smoothing at α=0.1 or 1.0 should stabilize frequent bigrams. For Q2, it’s clear that 16 million tokens can provide a solid warm start, especially if I do a dedicated pre-pass before training starts. I'll also consider document-boundary handling for SmearGate, making sure to mask the first position on chunks.Adjusting batch processing and bigram handling

If my training batches are packed sequences, the first position will still have the previous token from an earlier chunk. I should consider zeroing local features at that first position consistently. I might also explore using the exact bigram head as a residual of the hashed bigram, or even replace the bigram hash with a bigram head that includes count initialization and gating. If I'm over budget, reducing BigramHash buckets from 10,240 to 4,096 or removing it entirely when the exact bigram head is in use could work. Additionally, I should suggest storing the bigram head in int4 or using symmetric nibble packing for better compression.Finalizing bigram head quantization

I think the final quantization for the bigram head might be acceptable at int4, especially since it's just an additive logit bias, which could save me some space. I need to verify their artifact sizes; it’s currently 16.11 MB, over by 112 KB. If quantized to int5, it’s about 655 KB, so I might need to reduce one or two other tensors or lower the BigramHash buckets. Lowering the bigram_vocab_size from 10,240 to 8,192 or 6,144 could simplify things, too. Additionally, I should check their Q1 choices regarding BOS/EOS tokens and clarify the implications of using the ▁ pattern.Assessing token preprocessing options

I’m considering whether I can legally modify preprocessing to include an explicit EOS token per document in the training stream and handle that during evaluation. However, if the tokenizer is fixed with reserved BOS/EOS IDs, I might face limitations. The training artifact is self-contained, but can I change data preprocessing? Since the evaluation dataset is likely tokenized on the fly and we don't allow external downloads, it complicates things. I think it’s better not to assume. My answer would be to determine if explicit markers exist in the TokenStream and mask only position 0 and any present BOS/EOS tokens. In the future, I could look into modifying the tokenizer, but that would be scrutinized.Considering gamma values and quantization

I might suggest that the claim about "gamma values not stored in the artifact" is acceptable. However, if learned gamma affects quantization scales, the resulting row scales could differ, potentially impacting compressibility. It’s important to add a caution regarding the use of softplus/exp with explicit clipping bounds of [0.5,1.5]. If a hard clamp is used during the forward pass, there’s a risk of gradients at those bounds becoming ineffective, which could be problematic.Exploring count prepass and redundancy

I’m looking at the count prepass for training over the first M tokens. The code calculates pairs and the count matrix. I’m thinking about row centering and how to handle a sparse top-k residual. When selecting top values, it might be more effective to consider both the top positives and negatives, focusing on magnitude.

Also, I should mention the redundancy between BigramHash and the exact bigram head. If the table size is still an issue, my first recommendation is to modify or remove BigramHash(10240), stating clearly not to use both by default, as the bigram head could replace it. This change feels really important!Considering ablation and boundary handling

I need to consider whether to keep both options only if ablation shows an additive gain. I should mention "SmearGate boundary handling", following the same rules as bigram features—if there are no explicit boundary markers, only zero at the sequence start.

For the Q3 question about whether to separate offline steps versus a 10-minute wall clock, I think offline hyperparameter search is the way to go. It's robust across hidden seeds and clusters.

Also, discussing artifact size with low-rank bigram is essential. Quantizing the exact bigram table to int4 can save enough space, making it a straightforward solution!Fixing the bigram head process

In Step 1, I need to fix the bigram head, and the path list should include:

Quantizing the bigram table to int4.
Shrinking or removing BigramHash.
Addressing sparse top-k residuals.
Implementing low-rank factorization.
This would also address Q4. However, I realize that the bigram table requires count initialization; if it's quantized to int4, the training will still be in float. It's important to note that the bigram table adds to the logits, so low-rank inference costs shouldn't be a concern.

For Step 4, regarding sensitivity bit allocation, I think using candidates 4, 5, or 6 makes more sense, as 8 bits might be unnecessarily expensive. I might only allow 8 bits for especially sensitive tensors. Restricting to a shortlist can save time!Noting the BigramLogitHead class

I should mention that the "BigramLogitHead class is implemented but disabled" if the exact table exceeds the size limit. I can explore better compression since the count-initialized matrix has a strong row structure. Using delta coding across rows or sorting rows by token frequency might be helpful.

However, I must be cautious about function preservation since row order corresponds directly to previous token IDs, meaning I can’t permute rows or columns arbitrarily. I can consider using the difference from the sparse unigram vector per row, which reflects that sparse residual concept. I’ll ensure math formatting is clear with .... Time to craft the answer!

Yes — this is **much better**, but I’d still change a few important things.

## Biggest changes I would make

| Section | Change |
|---|---|
| **A1 / Step 3** | Fix the QAT pseudo-code: as written, it is **not** actually STE for weights. |
| **Bigram head plan** | Before low-rank, try **int4 for the bigram table**, **shrink/remove BigramHash**, or **sparse top-$k$ bigram residuals**. |
| **Q1** | `▁` is **not** a document boundary. Do **not** use that heuristic. |
| **Q2** | Count-init should happen in a **short pre-pass before training**, not mid-training if you can avoid it. |
| **Q3 / Step 4** | Sensitivity tests should compare against the **current quantized baseline**, not “others stay float”. |
| **Q5** | Remove `compression_reg()` from the mainline unless ablations prove it helps. |
| **Q6** | Do **temperature scaling**, but skip per-layer bias correction for now. |

---

## 1) The most important technical correction: your QAT code snippet

This part is the main thing I would change.

Your current sketch:

```python
q = torch.round(w / scale).clamp(-clip_range-1, clip_range)
return q * scale  # STE: gradient flows through
```

As written, this is **not** true STE for the weights. `torch.round()` has zero gradient almost everywhere.

### Better version

```python
def ste_round(x):
    return (x.round() - x).detach() + x

def fake_quant_with_gamma(w, theta, qn, qp):
    # gamma in [0.5, 1.5]
    gamma = 0.5 + torch.sigmoid(theta)

    # detach row stats; let gamma learn, not amax
    amax = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = gamma * amax / float(qp)

    u = (w / scale).clamp(qn, qp)
    q = ste_round(u)
    return q * scale
```

### Also define the range explicitly

For example:

- int5: $q_n = -16,\ q_p = 15$
- int6: $q_n = -32,\ q_p = 31$

or use symmetric $[-15,15]$, $[-31,31]$ — **but be consistent** with export and packing.

### Small wording tweak for the doc

Instead of:

> “Learn `γ_i` per tensor on top of existing row-wise scales”

I’d write:

> “Learn a bounded per-matrix clip multiplier $\gamma_i$ on top of detached row-wise `amax` scales during QAT.”

That is more precise.

---

## 2) Your bigram-head section is missing the simplest way to fit under 16 MB

If you are only over by **112 KB**, I would **not** jump straight to low-rank first.

## Better order for Step 1

### Try these first

1. **Quantize the bigram table itself to int4**
   - A full $1024 \times 1024$ table at int5 is:
     $$
     1024^2 \cdot \frac{5}{8} = 655{,}360 \text{ bytes}
     $$
   - At int4 it is:
     $$
     1024^2 \cdot \frac{4}{8} = 524{,}288 \text{ bytes}
     $$
   - Raw savings:
     $$
     131{,}072 \text{ bytes}
     $$
   That is already roughly your overage.

2. **Shrink or remove `BigramHash` if exact bigram logits are enabled**
   - These two features are partly redundant.
   - If exact bigram logits help more, cutting `BigramHash(10240)` to `4096` or removing it may be the cleanest swap.

3. **Sparse top-$k$ bigram residual instead of full table**
   - This is honestly a very good contest hack.
   - Most rows only need a few strong next-token biases.

### I would add this to the doc

> If `BigramLogitHead` is enabled, first ablate with `BigramHash` reduced or removed before trying low-rank factorization.

That is probably the most useful missing change.

---

## 3) Add one more option: sparse top-$k$ bigram residuals

This is the main “clever” thing I think you are still missing.

Instead of a dense $1024 \times 1024$ table, store only the strongest residuals per previous token.

Let:

$$
R_{a,b} = \log p(b \mid a) - \log p(b)
$$

Then for each previous token $a$, keep only the top-$k$ entries of $R_{a,:}$ by magnitude.

### Why this is attractive

- Bigram conditionals are sparse-ish.
- Most useful signal is concentrated in a handful of next tokens.
- Storage is tiny.

For example, with $k=8$:

- pairs stored: $1024 \cdot 8 = 8192$
- token ids fit in `uint16`
- values can be int8 / int4-ish or FP16

This can be **dramatically** smaller than a dense table and may outperform a low-rank factorization for the same bytes.

If you want a new Step 1 path, I’d write:

- **(a)** int4 dense table
- **(b)** reduce/remove `BigramHash`
- **(c)** sparse top-$k$ bigram residual
- **(d)** low-rank factorization

I would put low-rank **last**.

---

## 4) Q1 document boundary detection: change this section

This section needs tightening.

## What to change

### Definitely remove

- option **(b)** `▁` pattern  
  That is a **word boundary marker**, not a document boundary.

### My answer

- If the token stream has explicit BOS/EOS/doc separators, use them.
- If it does **not**, then you do **not** have reliable document boundaries.
- Do **not** invent heuristics like period/newline masking.

### What you should still do

Even if no document boundaries exist, always mask local features at **sequence position 0**:

- `BigramLogitHead`
- `BigramHash`
- `SmearGate`

because position $0$ has no valid previous token inside the current sequence.

### Suggested wording

> If `TokenStream` does not expose explicit document separators, skip document-boundary masking and only zero local features at sequence position 0. Do not use `▁`, punctuation, or newline heuristics.

That’s the correct version.

---

## 5) Q2 count-based initialization: do a short pre-pass

I would change this section pretty clearly.

### Best timing

Do a **small counting pre-pass before gradient training starts**.

You do **not** need all $10$B tokens.

A sample of roughly $8$M to $32$M tokens is enough for a useful warm start because:

- vocab is only $1024$
- the bigram head is trainable afterward
- you only need a decent prior, not a perfect model

### Better than mid-training re-init

I would avoid:

- training for $100$-$200$ steps
- then replacing the table

That is awkward and can destabilize things.

### Better recipe

1. Read first $M$ tokens from training stream
2. Count bigrams with `torch.bincount`
3. Build residual table:
   $$
   B^{(0)}_{a,b}
   =
   \log \frac{C_{a,b} + \alpha}{C_{a,\cdot} + \alpha V}
   -
   \log \frac{C_{\cdot,b} + \alpha}{N + \alpha V}
   $$
4. Clip it to a sane range, e.g. $[-4, 4]$
5. Initialize the trainable table from that

### Is $16$M tokens enough?

Yes, for initialization, I think that is **enough to be useful**.  
Not perfect, but useful.

---

## 6) Q3 sensitivity allocation: fix the experimental setup

This is another important change.

Your current Step 4 says:

> quantize tensor $i$ to bit $b$, others stay float

I would change that.

## Better baseline

Compare each candidate tensor-bit choice against the **current quantized baseline assignment**, not against float.

For example:

- baseline assignment = current mixed int5/int6 scheme
- candidate = flip one tensor from int6 to int5 or int5 to int4

Then measure:

$$
\Delta_{i,b}
=
L(\text{baseline with tensor } i \to b)
-
L(\text{baseline})
$$

That is much more relevant to the final artifact.

### Also: compressed byte cost is not perfectly additive

If you use one global zstd stream, then “per-tensor compressed bytes” is only an approximation.

So I’d write:

- use per-tensor compressed bytes for the first ranking pass
- then verify the final chosen assignment on the **full serialized artifact size**

### Online vs offline

For the actual submission path, I’d recommend:

- **offline** determine a robust default assignment across a few runs
- hard-code it
- maybe do a tiny online local search only on a shortlist if time allows

So for Q3, I would favor **(b)** as the default plan.

---

## 7) Q4 low-rank factorization: make it Plan C, not Plan A

I would not lead with low-rank.

### My answer

- $r=64$ is plausible but somewhat aggressive
- $r=96$ or $r=128$ is safer
- but if you are only over by $112$ KB, it is probably unnecessary

So I’d reframe the doc:

> Use low-rank only if int4, `BigramHash` reduction, and sparse residual variants fail.

That is the right ordering.

---

## 8) Q5 compression regularization: remove it from the mainline

I would simplify here.

Your current `row_range.mean()` regularizer is still heuristic and not clearly tied to actual artifact bytes.

### My recommendation

- **remove it from the default plan**
- keep it only as an optional ablation flag

A good rule would be:

> keep it only if it saves at least about $100$-$150$ KB with less than about $0.001$ bpb loss

Otherwise it is just noise.

So yes: I think it is basically **redundant / not worth mainline complexity** right now.

---

## 9) Q6 post-quant calibration: do temperature, skip layer biases

### Temperature
Yes, exactly:

$$
\text{logits}' = \frac{\text{logits}}{T}
$$

with $T > 0$ fit on a tiny calibration set after quantization.

I’d parameterize $T = e^\tau$ and optimize $\tau$.

This costs one scalar and is easy.

### Bias correction
For now, I would **not** add per-layer bias vectors.

Reasons:

- code complexity
- extra tensors in export
- uncertain payoff
- your model is bias-free already

If you want a second calibration step beyond temperature, I’d rather do:

- freeze weights
- optimize only the clip multipliers $\gamma_i$ for a few calibration steps

That is more aligned with your quantization setup.

So Q6 answer should be:

- **Yes** on temperature
- **No / later** on per-layer bias correction

---

## 10) One practical thing missing: the smoke-test table needs a disclaimer

Your table:

| Run | Features | Artifact | Fits? | val_bpb (200 steps) |

is okay internally, but I would label it more clearly.

Because at **200 steps**, differences like $1.9385$ vs $1.9482$ are not very predictive.

### Suggested change

Rename column to:

- `smoke val_bpb @ 200 steps (not predictive)`

or add a note:

> 200-step smoke losses are only sanity checks for code path and size; they are not used to rank feature quality.

That will make the doc look more rigorous.

---

## Suggested revised priority order

I’d make it:

1. **Enable BigramLogitHead in a size-fitting form**
   - first try int4
   - then reduce/remove `BigramHash`
   - then sparse top-$k$
   - only then low-rank

2. **Add gate + count-init**
3. **Zero local features at sequence position 0**
4. **Learnable clip multipliers with correct STE**
5. **Offline sensitivity-based bit allocation**
6. **Post-quant temperature calibration**
7. **Only then revisit trigram / recurrence**

---

<details>
<summary><strong>Concrete text edits I would make</strong></summary>

### Replace Q1 section with something like

```markdown
### Q1: Document Boundary Detection

If `TokenStream` does not expose explicit BOS/EOS/document separators, then we do not have reliable document boundaries.

- Do **not** use `▁`, punctuation, or newline heuristics; those are not document boundaries.
- Always zero local features at sequence position 0.
- Only apply boundary masking at true BOS/EOS/doc separators if they exist in the tokenized stream.
```

### Replace Step 1 with

```markdown
### Step 1: Fit BigramLogitHead under 16MB

Need ~0.112 MB of headroom. Try in this order:

- **(a)** Quantize the bigram table to int4
- **(b)** Reduce or remove `BigramHash` when exact bigram logits are enabled
- **(c)** Sparse top-k bigram residual per previous token
- **(d)** Low-rank factorization `B = U @ V` as a fallback
```

### Replace Step 3 code with

```python
def ste_round(x):
    return (x.round() - x).detach() + x

def fake_quant_with_gamma(w, theta, qn, qp):
    gamma = 0.5 + torch.sigmoid(theta)  # gamma in [0.5, 1.5]
    amax = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = gamma * amax / float(qp)
    u = (w / scale).clamp(qn, qp)
    q = ste_round(u)
    return q * scale
```

### Replace Step 4 wording with

```markdown
For each tensor `i` and candidate bit-width `b`, measure loss delta relative to the current mixed-precision baseline assignment, not relative to an all-float model. Use per-tensor compressed size as a first-pass proxy, then verify the final chosen assignment on the full serialized artifact.
```

</details>

## Bottom line

Your writeup is now **mostly solid**, but I would definitely change:

- the **QAT pseudo-code**
- the **boundary section**
- the **bit-allocation baseline**
- the **Step 1 bigram-fit plan**

And I would add one missing option:

- **sparse top-$k$ bigram residuals**

If you want, I can turn this into a **clean Round 2 reviewer memo** you can paste directly, with the corrected wording already baked in.