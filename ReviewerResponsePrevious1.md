Addressing user questions

I need to respond to the user’s five questions thoughtfully, ensuring I provide corrections and actionable guidance.

I should answer the question about what they should ask the reviewer. Since I’m playing the role of the reviewer, I’ll directly address aspects related to BigramHash. I’ll validate each idea presented, discuss implementation details, outline potential risks, and even reference relevant formulas.

And if it helps, I could include some code snippets to enhance clarity!Discussing implementation details

I might consider including formulas for least squares (LSQ) gradients, but I want to keep things simple. It could be useful to mention implementation specifics, especially since they pointed out that there’s “one float32 per quantized tensor = 40 params total” if there are 40 tensors.

I need to note whether per-row scales are already in use; using one scalar per row could reduce size overhead. If the current quantization uses per-row amax scales, then each quantized row is already storing a scale, which is important. The new step size could match these scales or use learned versions without additional overhead.Clarifying quantization details

The user mentioned that one float32 per quantized tensor equals around 40 parameters, but if the current scheme uses per-row scales, the learning focus might need to shift to per-tensor or per-row multipliers. I need to clarify that “one per tensor” might be too broad if quantization is already happening row-wise.

I think recommending a "learnable clip multiplier on top of per-row amax" during QAT could be beneficial. This would add just one scalar per tensor while utilizing existing row-wise scales. It addresses their risk nicely! Learning a full scale per tensor wouldn’t fit current row-wise deployment unless the serialization scheme is changed. Thus, a per-tensor clip multiplier keeps compatibility.Examining quantization strategies

If their deployment uses row-wise serialization for int5/int6 scales, then applying a full LSQ single scale per tensor might not fit well since inference relies on row scales. A better approach could be to use a per-tensor or per-group clip factor that multiplies each row's amax or percentile.

I’m considering whether to learn either γ or the percentile, and possibly logγ with a regularizer.

Also, I need to differentiate between PACT for activations and LSQ for weights, noting that their training likely only quantizes weights. If the inference artifact only quantizes weights, activation quantization might not be required. However, regularizing through activation quantization during training could still be beneficial, though probably not essential. For weights, we could implement learnable quant clipping, leaving activation clipping optional unless low-precision activations are involved. Finally, I want to think about the exact bigram logit head if there’s a 1024 vocabulary size, resulting in 1,048,576 entries.Analyzing BigramHash and compression

The int5/int6 compressed size might be around 0.6-0.8MB, but I think I need to correct that to about 0.5-0.65MB raw at int4/int5 if they're tied. Calculating further, 1,048,576 times 0.625 bytes gives 655,360 bytes at int5 raw. Considering per-row scales adds around 2KB, so that’s manageable. It may also be better than the trigram hash.

For gating, I’m suggesting something like logits=LM(h)+αB[x 
t−1
​
 ], where α is a learned scalar or context-dependent gate. Using low-rank factorization could help if we're over budget, and resetting boundaries is an option too.

Combining with unigram prior or Kneser-Ney smoothing might be too complicated. Regarding BigramHash, I should note that the existing SOTA is pretty solid; my main concern lies with the trigram formula. I could mention that there are better mixers but no evidence that the current one is fundamentally flawed. Mathematically analyzing it gives me h(a,b)=(c 
1
​
 a⊕c 
2
​
 b)modm.Exploring XOR and hash functions

I’m thinking about how ⊕ operates on bit patterns once multiplication is applied, especially since it's modulo non-power-of-two. This means we can't exactly eliminate one token like we would in a packed scheme. Given that the multipliers are odd, the mapping from a to c 
1
​
 a appears to be injective over integers and seems to preserve lower bits mod 2 
k
  as permutations.

XORing with an independent term functions as a bijection on 32-bit words for fixed b. Reducing mod m introduces some collisions, but it doesn't systematically drop tokens, so that's okay. A final avalanche could enhance distribution a bit if the table size is large. I’m suggesting using prime buckets near the desired size, perhaps with 64-bit mixing, and I should consider including sample code for robust hash functions.Including hashing code examples

I should include a Python code snippet for the bigram and trigram hashing, like this:


def mix32(x):
    x ^= x >> 16
    x *= 0x7feb352d
    x ^= x >> 15
    x *= 0x846ca68b
    x ^= x >> 16
    return x

h = mix32(a * 0x9e3779b1 ^ b * 0x85ebca77) % M
It's important to remember that in Python, we need to mask to 32-bit for accurate results. If using PyTorch with int64 operations, there’s no overflow wrapping. We might need to cast to torch.int64 and apply a bitmask using & 0xffffffff for exact mixing.

Alternatively, I could simplify using a linear congruential mixture with a prime modulus M, or implement universal hashing with a formula like ((aA+bB+cC)modP)modM, where P is a large prime. This avoids bitwise operations and is vectorizable, an example being P=2 
61
 −1.

Considering their risk claims, I need to address learnable clipping, particularly the risk of scale drifting. I might suggest a parameterization that keeps values positive and narrows around the initial scale.Discussing scale and training adjustments

I'm considering using γ=exp(δ) with δ clamped within the range [log0.25,log4], or possibly implementing a sigmoid function around amax. Another option is to regularize (logs−logs 
0
​
 ) 
2
  to keep it small, but clamping too aggressively near amax might not yield benefits. It’s likely better to initialize from the current scale and apply gentle constraints.

I also need to clarify that for the model's forward function, it's better to return per-token losses separately from the training loop's manual weighting. Keeping the model clean is essential; we should use reduction='none' during training.

Accessing a byte lookup table (LUT) will probably simplify things, and I'll need to address handling padding masks and packed sequences. We can weight by the target token's byte length and normalize that by the sum of weights over non-masked tokens. If document packing is involved, they'll have to ensure the LUT includes both BOS and EOT tokens correctly.

Lastly, noting the difference between bits and nats: if using F.cross_entropy with natural log, the bits per byte (bpb) objective can be represented as  
ln2
1
​
  
∑bytes 
t
​
 
∑ℓ 
t
​
 
​
 . Since dividing by that constant doesn't change the outcome, it's sufficient to focus on weighting by bytes.Proposing sensitivity-based bit allocation

I'm thinking about developing a precise calibration protocol for sensitivity-based bit allocation. For each tensor (i) and bit candidate (b), I need to compute (\Delta_{i,b}) on the calibration set by quantizing just tensor (i) to (b) while keeping the others in floating-point or baseline quant. I’ll also compute the byte cost (c_{i,b}) and then optimize accordingly.

If the baseline is already mixed, it’s better to compute the marginal delta relative to the current bit assignment. A greedy approach could work here, either starting all tensors at max bits and downgrading low-sensitivity ones until the budget is reached, or starting at min bits and upgrading the ones with the highest benefit per byte.

Since the candidate set is small, dynamic programming might be possible, but the greedy ratio may suffice. I should mention how the interactions between tensors mean independent estimates are just an approximation. To address this, a short QAT or fine-tune after the assignment might be necessary.

Also, I’m considering that using "one forward pass per tensor" could be noisy. Implementing 8-32 mini-batches and evaluating validation NLL rather than training could be more effective. But I wonder, is it feasible to keep this process under 10 minutes?Considering additional calibration techniques

I’m thinking about the importance of pre-computing after training on a small held-out calibration set. It might be valuable to mention a low-risk extra step: applying "per-layer temperature/bias correction" after quantization. This mirrors the GPTQ/AWQ style bias correction for each linear layer to offset any quantization error. Since it's a smaller model, calibrating layer outputs seems manageable.

While it may be an extra detail, it aligns with the user’s inquiry about simple fixes or clever hacks. I can suggest two extras: boundary-aware local heads and post-quant bias correction.

I need to keep in mind the challenge rules they've set, especially regarding no external downloads. I’m thinking of structuring my response with sections for each item, including verdicts, corrections, and specific implementation advice. I could also provide a ranking like, "If I had to prioritize: 1 > 3 > 4 > 5 > 2."Evaluating hash and bigram implementations

I need to reconsider the hash if an exact bigram is already implemented; maybe I should rank it lower unless the current BigramHash is still being used. My ranking could be: learnable clip, byte loss, exact bigram head, sensitivity-based bit allocation, and hash fix for trigram. If the exact bigram head runs, the BigramHash fix might not be necessary.

I should mention that if the exact bigram logit head is already in place, the BigramHash could be redundant or could coexist. It’s worth comparing four variants: baseline, BigramHash only, exact bigram bias only, and both with a learned scalar gate.

There may be potential redundancy since BigramHash feeds hidden states, while the exact bigram head contributes direct logits. I need to be cautious about stacking too many local heuristics without gating. Additionally, if they have a BPE vocabulary of 1024, using an exact bigram bias table brings a first-order Markov residual, which pairs well with byte-normalized loss and is cost-effective.Exploring initialization strategies

I could initialize from empirical counts derived from the training data and then fine-tune. The challenge mentions that the artifact includes code and weights, so I can precompute these counts during training. This might be a clever hack: using a smoothed count bigram table from the training corpus instead of learned parameters. Since the evaluation dataset comes from the same distribution, methods like Kneser-Ney or add-α counts should perform well.

If training time is limited, counting bigrams is relatively simple. It’s worth considering a hybrid approach that combines neural logits with a Kneser-Ney bigram model. Since there’s no external data involved and the metric is based on FineWeb, this count-based n-gram model could be an effective local model.

If I keep it under 16MB, I might store the top bigrams or a compressed bigram table, as 1024x1024 counts become small when quantized. Additionally, since the user already has an exact bigram logit head, I could suggest initializing that table from smoothed empirical log-probs instead of using random or zero values. This could accelerate training in about 10 minutes! It's a great idea, and I’m excited to develop it!Analyzing count-based models

For the trigram head, creating an exact trigram table is impractical due to the size—that's 1024 
3
 , which is enormous! Instead, I could consider using low-rank or hashed representations and perhaps store the top K count-based trigrams. However, this may be out of the project’s scope. I also want to mention "boundary masking at document starts" since n-grams are sensitive to context. Evaluating on the first 50k documents and resetting local state for each document is crucial to maintain quality.

If training packs documents without this consideration, including cross-document bigrams might lead to poor learning. This is an important point to address. I could suggest that when answering the reviewer, I mention that I don’t think the current BigramHash is broken, but the issue lies primarily with the trigram formula. I would still recommend using a better mixer for the trigrams and possibly adding a final avalanche for the bigrams.Evaluating improvements

Regarding hash mixing, the current bigram hash seems fine, but my concern is with the naive trigram formula. Upgrading to a stronger mixer might help, but the impact may not be significant.

For byte-normalized loss, weighted-by-bytes isn't effective. It's better to use standard token Cross-Entropy. If exact bits per byte scaling is desired, simply divide the batch sum by the total bytes, as this only affects the batch scalar, making it a low priority.

The exact bigram logit head is strong, but I have some recommendations: I could add a learned gate α, reset at document boundaries, and initialize from smoothed counts. Depending on size, staying within int5/int6 or even FP16 with row-wise quantization could work well. Lastly, I could consider tying it with count-based initialization.

For sensitivity-based bit allocation, using true serialized bytes is an excellent approach, following the steps mentioned before.Addressing suggestions and corrections

There are a couple of simple things that might be missing: first, document-boundary masking for all n-gram features and cache heads, and second, count-based initialization or interpolation for the bigram head. Also, incorporating post-quant calibration including temperature and bias correction can be helpful. Using true compressed bytes in the knapsack might also improve performance.

It’s important to correct earlier advice. Specifically, the byte-normalized formulation I mentioned isn't the right per-token weighting. The existing bigram hash with XOR primes modulo 10239 isn't broken — I'd consider it "okay but improvable." While there are collisions due to the modulo, the mapping remains roughly uniform.Exploring hash options and adjustments

I could point out that a bucket count of 10239 feels a bit awkward, and using a prime like 12289 might make a difference. If the exact bigram head is already implemented, perhaps the BigramHash could be retired if it's redundant. However, the bigram hash improves the hidden representation, while the exact bigram bias adjusts logits directly, potentially dominating the outcomes.

Should I mention that SOTA uses STE QAT with fake int5/int6 using amax per row as scale? We’d need to clarify that if deployment stores per-row scales, it changes training but not the final quantization, creating a mismatch. The cleanest solution might be to learn a tensor-specific clip multiplier to scale the row-wise amax or percentile.Refining initialization and scaling approaches

I should discuss using equations to initialize scales, either via max values or percentiles, which are cheap and effective. We can learn row-wise scales directly since they're stored, minimizing artifact costs. It’s important to note that quality could drop if learned scales drift too far. To prevent this, we might initialize them at current calibration and unfreeze at QAT.

An interesting hack would be to use empirical bigram counts to initialize the exact bigram table, allowing for immediate local-statistics prior without needing to learn from scratch. Additionally, mentioning sensitivity-based bit allocation can highlight its novelty in this context.

I can clarify about document-boundary handling and recommend calibration modifications. Finally, if we combine these elements, we might rank new experiments focusing on the exact bigram head with gating, post-quant calibration, and more. I could offer to provide code patches for the implementation if requested!Evaluating forward pass efficiency

I think it’s essential to consider that for #5, we could do "one forward pass per tensor" with a focus on "loss-only forward, no backward," which would be more economical. With 30 tensors and multiple bitwidths, performing about 720 forwards could be manageable, especially if the model is trained for less than 10 minutes. However, any extra calibration time will need to fit within that limit.

To make it more efficient, I might suggest computing on just 1-2 calibration batches or using a Hessian approximation, focusing only on borderline tensors after starting from a baseline.Considering calibration efficiency

I realize that doing 720 forward passes could be too much within the 10-minute limit if the training is already saturated. A good approach might be to calibrate on a very small held-out subset or during the last few hundred training steps. If there are 40 tensors and 3 bitwidths, that totals 120 extra passes, which might not be trivial, especially with a sequence length of 2048 on an 8xH100 setup. It could still work if I only use 1-2 batches and don't overdo it. Also, I'd want to ensure that I use training data for calibration to avoid overfitting. Lastly, although mixed precision bit allocation isn't new, the contest’s specific objective with post-zstd byte costs is a clever twist!

Yes — this is a much sharper list. But **one of the five is conceptually wrong**, and a couple need narrowing.

## Short verdict

| # | Idea | My verdict | Priority |
|---|---|---|---|
| 1 | Learnable quant clipping | **Yes, strong** | **High** |
| 2 | Fix existing BigramHash mixing | **Current one is probably fine**; concern was mainly your trigram formula | Medium-low |
| 3 | Byte-normalized loss | **As you described it: no, that weighting is wrong** | **Drop / very low** |
| 4 | Exact bigram logit head | **Yes, very good** | **High** |
| 5 | Sensitivity-based bit allocation | **Yes, one of the best ideas here** | **High** |

The biggest correction is **#3**.

---

## 1) Learnable quant clipping

**Your direction is right.**  
But I would implement it slightly differently from “one LSQ step size per tensor” if your final packing is already **per-row scaled**.

### The key distinction

If your deployed quantization is something like:

$$
q_{r,j} = \mathrm{clip}\left(\mathrm{round}\left(\frac{w_{r,j}}{s_r}\right), -Q, Q\right)
$$

with a stored scale $s_r$ per row, then a pure **single learned tensor-wide LSQ scale** does not match the final artifact very well.

### Better drop-in version

Keep your rowwise scaling, but learn a **small clip multiplier per tensor**:

$$
s_r = \frac{\gamma_i \cdot a_r}{Q}
$$

where:

- $a_r$ is the current row statistic, e.g. `amax(row)` or row percentile
- $\gamma_i$ is a learned scalar for tensor $i$
- $Q = 15$ for signed int5, $Q = 31$ for signed int6

This gives you almost all the benefit with tiny code changes and no mismatch at export.

### Why this is better than hard-clamping to `amax`

Your instinct about drift risk is right, but I would **not** hard clamp it tightly to `amax`. That kills the point.

Instead:

- parameterize $\gamma_i > 0$ with `softplus` or `exp`
- initialize at $\gamma_i = 1$
- constrain gently, e.g. $\gamma_i \in [0.5, 1.5]$ or regularize $\log \gamma_i$

### Practical recipe

- Start QAT at the same point you already do
- Unfreeze $\gamma_i$ only during QAT
- Give $\gamma_i$ a lower LR than weights
- Optionally regularize toward $1.0$

A simple form:

$$
L = L_{\text{task}} + \lambda \sum_i (\log \gamma_i)^2
$$

### My verdict on your “size cost”

Yes, essentially zero.  
And if you already store row scales, the final artifact overhead is basically unchanged.

### Bottom line

**Yes, do this.**  
But I’d phrase it as:

> “Learn a per-matrix clipping multiplier on top of existing rowwise scales,”

not necessarily “full LSQ per tensor replacing current row scales.”

---

## 2) Existing BigramHash mixing

Here’s the direct answer:

> **I do not think the current SOTA BigramHash is catastrophically broken. The earlier concern was mainly about your naive trigram formula.**

### Why the current bigram hash is probably okay

You said the current hash is:

```python
h = (36313 * t1) ^ (27191 * t0)
idx = h % 10239
```

That is very different from the bad packed form:

```python
idx = (t0 * 1024 + t1) % M
```

The bad packed form can systematically discard information when $M$ and $1024$ interact badly.

The XOR-multiplier form does **not** have that same pathology. Before the final modulo, both tokens affect the word in a nontrivial way.

So:

- **not broken in the same way**
- probably “good enough”
- maybe improvable with a final avalanche mix, but not urgent

### If you want a safer hash anyway

Use something like:

```python
def mix32(x):
    x = x ^ (x >> 16)
    x = (x * 0x7feb352d) & 0xffffffff
    x = x ^ (x >> 15)
    x = (x * 0x846ca68b) & 0xffffffff
    x = x ^ (x >> 16)
    return x

def hash2(a, b, m):
    x = ((a * 0x9e3779b1) ^ (b * 0x85ebca77)) & 0xffffffff
    return mix32(x) % m
```

But I would treat that as a cleanup, not a must-have.

### My recommendation

- **For existing BigramHash**: probably leave it unless profiling says otherwise
- **For TrigramHash**: definitely use a real mixer, not packed modulo

---

## 3) Byte-normalized loss

This is the one I would **correct strongly**.

### What you wrote is not the right objective

You wrote:

> Train on $\sum_t \ell_t / \sum_t \text{bytes}_t$ instead of $\sum_t \ell_t / T$, and give tokens with more bytes more weight.

The first half is fine.  
The second half is **not**.

### Correct derivation

Let $\ell_t = -\log p(y_t \mid \text{context})$ in nats.

Then the true bits-per-byte objective for a batch is:

$$
L_{\text{bpb}} = \frac{1}{\ln 2} \cdot \frac{\sum_t \ell_t}{\sum_t b_t}
$$

where $b_t$ is the byte length of target token $y_t$.

Crucially:

- the denominator $\sum_t b_t$ is a **constant for the batch**
- so the gradient is proportional to:

$$
\nabla L_{\text{bpb}} \propto \sum_t \nabla \ell_t
$$

not

$$
\sum_t b_t \nabla \ell_t
$$

### So what does that mean?

It means:

- **do not weight each token loss by its byte length**
- that would optimize the wrong objective

If you switch from token-mean CE to exact batch bpb, the only change is a **batch-level scalar rescaling**:

$$
\frac{1}{T}\sum_t \ell_t
\quad \to \quad
\frac{1}{\sum_t b_t}\sum_t \ell_t
$$

If your batches always have the same number of tokens, this is basically just a tiny adaptive learning-rate tweak based on average bytes per token.

### Practical conclusion

I would **not prioritize this**.  
Maybe keep it in mind as a “nice exactness cleanup,” but I would not expect a meaningful gain.

### So the correct answer to your implementation question is

- You do **not** need per-token byte weighting
- If you really want exact batch bpb scaling, use `reduction='sum'` and divide by total bytes in batch
- But I would rank this very low

---

## 4) Exact bigram logit head

This is still one of my favorite ideas in your whole stack.

## Why it’s strong

With vocab size $1024$:

$$
1024 \times 1024 = 1{,}048{,}576
$$

That is small enough to be practical.

And unlike hashed bigram embeddings:

- no collisions
- direct effect on logits
- very local web-text patterns get captured cleanly

### Best way to use it

Not just:

$$
\text{logits}_t = \text{LM}(h_t) + B[x_{t-1}]
$$

but:

$$
\text{logits}_t = \text{LM}(h_t) + \alpha \, B[x_{t-1}]
$$

with a learned scalar $\alpha$ or even a tiny gate.

That lets the model decide how much to trust the bigram table.

### Two very important details

#### A. Reset at document boundaries

This matters a lot.

If you pack documents back-to-back in training, then the transition from the last token of doc $A$ to the first token of doc $B$ is fake.  
Your bigram head and hashed n-grams should **not** see that as a real bigram.

So you want a boundary mask:

```python
bigram_bias = B[prev_tokens]
bigram_bias = bigram_bias * (~is_doc_start).unsqueeze(-1)
```

This is one of the easiest hidden mistakes in local-feature hacks.

#### B. Initialize from smoothed counts if you can

This is a genuinely good contest hack.

Instead of starting `B` from zeros or random, initialize from corpus counts:

$$
B_{a,b} \approx \log p(b \mid a)
$$

or better as a residual over unigram frequency:

$$
B_{a,b} = \log p(b \mid a) - \log p(b)
$$

Then the bigram table starts as a real count-based language model and training only needs to refine it.

That can matter a lot under a 10-minute budget.

### My verdict

**High priority.**  
Honestly, I would test this before trigram hash.

---

## 5) Sensitivity-based bit allocation

This is excellent. Probably better than most architecture changes.

### Core idea

For each tensor $i$ and candidate bitwidth $b \in \{4,5,6,8\}$, estimate:

- quality hit: $\Delta_{i,b}$
- storage cost: $c_{i,b}$

Then solve:

$$
\min_{\{b_i\}} \sum_i \Delta_{i,b_i}
\quad \text{subject to} \quad
\sum_i c_{i,b_i} \le 16\text{MB}
$$

### One very important improvement

Do **not** use only nominal bit cost like:

$$
c_{i,b} = \frac{b}{8} \cdot \#\text{params}
$$

If your contest is won or lost by `zstd`, use the **true serialized-and-compressed bytes** as the cost.

That is the clever version.

Different tensors compress differently:

- some int5 tensors zstd amazingly well
- some int5 tensors compress poorly and are almost as expensive as int6
- scales / metadata also matter

So define $c_{i,b}$ as:

> actual bytes after packing tensor $i$ at bitwidth $b$ and compressing it the same way you will in the artifact

That is much more contest-aligned.

### Your “one forward per tensor” plan

Reasonable first approximation, but I’d refine it slightly:

- use a small calibration set, maybe 2 to 8 batches
- evaluate a few candidate bits per tensor
- compute marginal loss from replacing the current bit assignment with candidate $b$

Then either:

- do dynamic programming if tensor count is small
- or use a greedy ratio based on “loss increase per saved byte”

### One caution

Tensor sensitivities interact.  
Quantizing two tensors together can hurt more than the sum of each separately.

So I’d do:

1. independent estimate
2. choose assignment
3. short final QAT / finetune with that assignment

That should absorb most interaction effects.

### My verdict

**Very high priority.**  
Not novel in the literature, but very good for this contest.

---

## The one thing I would actively change in your plan

I would replace:

> Byte-normalized loss

with:

> **Count-initialized exact bigram head + document-boundary masking**

That is more likely to move the metric.

---

## What you may still be missing

### 1. Document-boundary handling for every local trick

This applies to:

- exact bigram head
- BigramHash
- TrigramHash
- cache-like tricks
- recurrence state if any state leaks across docs

This is small but can absolutely matter.

---

### 2. Post-quant calibration after training

Very cheap, often useful:

- fit final clip multipliers on a small calibration set
- optionally fit a global logit temperature after quantization
- maybe simple per-tensor bias correction if your linear layers have biases or can absorb an offset

This is much lower risk than recurrence.

---

### 3. Count-based initialization is a real “hack”

For the bigram head, initialize from empirical smoothed log-probs.  
That gives you a good count LM for almost free.

This is one of the best “clever but legal” tricks here.

---

### 4. Use true compressed size inside the optimizer loop for bit allocation

This is the contest-specific version of mixed precision.  
It’s better than raw bit counts.

---

## My updated priority order

1. **Exact bigram logit head**
   - with learned gate
   - with doc-boundary reset
   - ideally count-initialized

2. **Learnable clip multiplier over rowwise quant scales**

3. **Sensitivity-based bit allocation using true compressed bytes**

4. **Post-quant calibration**

5. **Trigram hash only after using a proper mixer**

6. **Recurrence later, if ever**

---

## Direct answers to your numbered questions

### 1. Learnable clipping
**Yes.**  
Best form: learn a tensor-level multiplier on top of your existing rowwise scale scheme.

### 2. Existing BigramHash broken?
**Probably not.**  
My earlier concern applies mainly to the naive trigram formula, not your current XOR-prime bigram hash.

### 3. Byte-normalized loss
**As byte-weighted per-token CE: no.**  
That is not the right bpb objective.

### 4. Exact bigram logit head
**Yes, strong.**  
Add a gate, boundary masking, and preferably count-based init.

### 5. Sensitivity-based bit allocation
**Yes, excellent.**  
Use actual compressed bytes, not just nominal bits.

---

If you want, I can next give you either:

1. a **minimal `train_gpt.py` patch plan** for the exact bigram head done correctly, or  
2. a **clean implementation sketch for learnable clip multipliers** that matches rowwise int5/int6 export.