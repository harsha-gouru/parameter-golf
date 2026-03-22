Thinking through constraints

I need to respond to the user who’s asking for suggestions and code snippets while considering their context. They provided the current leaderboard and pending PRs for review, which might not be stable. I could base my suggestions on what they provided without needing to verify current information through the web, as the user didn't explicitly ask for the latest updates. It’s essential to be cautious about previous corrections, like the byte-normalized loss, and perhaps suggest new techniques.Analyzing constraints and suggestions

I need to think about constraints, like the 16MB artifact limit, 10 minutes on 8xH100, and the potential lines of the training script. The user mentioned using a function in their code that might be competitive, so I should suggest code changes accordingly. I might recommend changes like using LeakyReLU or FlashAttention to optimize performance. It's essential to evaluate gains from certain features and think quantitatively about options, especially since user’s additions may have negatively impacted their results.Examining significant differences

I’m analyzing the performance difference between two models: one has 6225 steps and the other, 7051, which is a significant 13% fewer steps. This disparity suggests more is going on than just the number of steps, possibly related to their implementation of components like XSA. I think I should suggest they replicate a proven model stack before trying something novel. There’s also a potential for adding an 11th layer within their headroom budget, which could enhance performance while considering compute costs.Considering pending PR implications

The pending PR seems promising since it achieves 7051 steps at 85ms, possibly due to faster overall implementation. For the user, if the 10-layer setup takes around 96ms, adding an 11th layer might push it to 105-106ms with about 5600 steps. An extra layer could improve bits per byte but with some trade-offs in step costs. I need to evaluate using EMA and SWA together while weighing code complexity. Using EMA throughout might simplify things while maintaining efficiency, especially on GPU despite potential memory costs.Evaluating EMA implementation and recommendations

I'm considering whether to update after each optimization step only on the master. Even though DDP weights are the same across ranks, it might be easier to keep the EMA state on each GPU. Using torch._foreach could add a slight delay of 1-2ms, but that's manageable. I also want to think through the differences between the bigram head and bigram hash. Since top PRs use BigramHash, I recommend the user disable the exact head in favor of BigramHash to save on time and space.Analyzing training step performance

I’m considering whether the results are comparable, especially since they might not account for quantization yet. Tracking improvements, I see that from step 5000 to 6225, there's a notable improvement of 0.0416 over 1225 steps, indicating they're still learning quickly at this stage. Additional steps could yield more gains. However, step time is critical, and sacrificing 4ms per step could significantly impact the results. Thus, the focus should definitely be on increasing speed for better performance!Emphasizing optimization strategies

It seems the top pending PR might achieve 11 layers at 85ms due to not using an exact head and optimizing the last four layers with XSA. The user's gap of 0.03 could primarily be from steps, so cutting the exact head and any overhead is crucial. To reach a performance goal of 1.1248, I should suggest closely matching the features from PR #315/#401, utilizing LeakyReLU^2, and reconsider the bigram bucket size along with possibly removing unnecessary components like the U-Net skip or SmearGate.Evaluating optimization options

I’m considering that the current top pending PR (1.1243) is likely superior due to the use of XSA on a simpler GPT model. There seems to be something fundamentally wrong with the user’s setup at 1.1551, possibly due to elements like relu² or the bigram head adding compute costs. Thinking about FlashAttention 3, it might be beneficial, but it’s unclear if it can be imported under the user's constraints. Using built-in options may be necessary for the artifact to remain self-contained.Reviewing potential optimizations

I should check if PR #315 mentions FlashAttention 3, ensuring it’s just PyTorch using the Hopper kernel, since an external package would likely be disallowed. There's also the idea of using a shared value embedding dimension of 128 in later layers. This could help save parameters and may speed up computation, but it isn't a priority given the existing headroom. I should suggest post-quantization temperature calibration, which is a simple adjustment to improve confidence metrics after quantization—this is a good, low-code approach. Also, sensitivity-based bit allocation might be less urgent at this stage until bit-packing is implemented.Exploring optimization options

If only the bigram table is packed in int4 while other tensors remain int8, they might need to consider sensitivity allocation, especially with zstd. Moving some large tensors to int4 could help, and packing all MLP tensors as int5 might save space and accommodate an 11-layer model with an exact head. However, if the exact head is removed, size should be adequate. The exclusive self-attention code might be costly due to F.normalize and repeat_interleave during each forward pass, so optimizing with a RMS norm-like reuse could be beneficial. I should also verify that Rotary.inv_freq size matches the correct dimensions for Partial RoPE, and explore potential fusions for LN Scale. Finally, considering gradient accumulation or tuning batch tokens could also enhance performance.Considering batch size and efficiency

It seems like the top PR might be using a different batch size that affects step time. They both have 85ms, but maybe the grad accumulation is lower. The wall clock time could be influenced by needing to use 8 GPUs. I wonder if the user should increase the learning rate with a smaller global batch. Reducing the batch from 786K to 655K could provide more steps in ten minutes while still improving loss. This might be an optimal batch size, especially since there’s a time and compute tradeoff to consider. I should mention testing different TRAIN_BATCH_TOKENS sizes like 524288, 655360, and 786432!Analyzing optimization strategies

It looks like bits-per-byte depends on the total tokens trained and optimization dynamics. I wonder if more steps at a smaller batch size could be beneficial under the same wall clock time since loss is still dropping. That's a low-code, high-ROI suggestion! I think a warmdown schedule of 40-45% might be good, considering we already have a 45% ratio with 2800 steps out of 6225. Using EMA updates with a foreach loop can be efficient! Regarding the 11L model, I need to consider size and compute costs carefully. If we keep the exact head, maybe avoid 11L, but if we drop it and use BigramHash (2048), then going for 11L could work!Suggesting code improvements

I should suggest some useful code snippets! First, I could implement efficient EMA and provide a temperature scaling calibration function. I’ll also consider adding a line for LeakyReLU² and incorporating smaller batch sweeps or batch size scaling. For the 11L configuration, I could look at switching the exact bigram head to a bigram hash or a hybrid gating system, possibly with code to disable based on the environment.

I’m wondering if using a smaller XSA setup, like only the last two layers, could save compute since the last four layers add time. Top PR uses XSA4, but if you're slower, maybe XSA2 or sharing vn repeat is worth considering.

Lastly, using Partial RoPE variations and ensuring FlashAttention 3 setups work correctly might be good ideas, utilizing environmental flags if necessary.Optimizing code considerations

I’m pondering some optimization strategies. It seems that PR #315 suggests that QAT might be dead code because of compile optimizations. This makes me think that other settings, like use_xsa or rope_dims, should be constant booleans for compilation to avoid per-layer dynamics.

I could suggest using fused RMSNorm, as they utilize F.rms_norm, and think about replacing repeat_interleave in the XSA with reshape/expand. This could be faster, so I’ll check that.

Also, it might make sense to avoid computing F.normalize(v, dim=-1) since using F.rms_norm or an equivalent could streamline it. Additionally, precomputing layer-scale constants instead of calculating at runtime might help, though it seems minor.

Lastly, I should mention that the "count-init prepass" doesn’t impact per-step time significantly—just startup. This is an important correction to note because it highlights the user’s included overhead isn’t directly responsible for the time differences.Analyzing bigram head optimization

I’m realizing that the bigram head's 4ms overhead raised my suspicions; its table lookup has a shape of [batch*sequence, vocab]. It seems that this head returns a full vocabulary bias table for previous tokens, resulting in a large read of 1024 floats per token, which could explain the delay.

To optimize, I’m considering low-rank approaches or storing the table in fp16/bf16 during training since inference uses float32. It seems like storing in bfloat16 could help reduce bandwidth, but the optimizer might need to stay in FP32. Overall, it looks like the cost arises primarily from the shape rather than the data type.Exploring bigram head optimization

I’m considering whether using a low-rank factorized bigram head during training might reduce compute, but I realize it could lose some accuracy, which probably isn't worth it.

I wonder if it could be beneficial to use the bigram head only in the first or last layers, but it's really tied to logits.

Instead, I’m thinking about using a sparse top-k bigram residual during training and inference to reduce computation. By adding only k next-token biases per previous token, I could use scatter_add on logits for the top k entries. This could approximate the exact bigram head while being more efficient. It seems like a clever hack that could cut parameters significantly!Evaluating bigram optimization methods

I’m looking at how count-init yields sparse strong biases, which might even dominate the BigramHash. That seems like a great suggestion! I think I should mention it as an alternative to the exact head in my code snippet.

For each previous token a, I'll store topk_idx[a,k] and topk_val[a,k]. During the forward pass, I can gather the indices and values for previous tokens and use logits.scatter_add_(1, idx, gate*val) before applying the softcap. This approach costs O(Nk) instead of O(NV), which could result in huge performance improvements.

It looks promising because it may retain most bigram signals while being faster and more compact. Considering the vocabulary size and total cost appears to be under 100KB, that’s rather tiny!Analyzing compute overhead and optimization

I’m evaluating compute overhead, particularly the scatter_add method over [N,1024] logits with k=16 values. It seems this approach should be much more efficient than loading a full 1024-row table for each token—definitely an important technique!

I need to provide a code snippet demonstrating how to build the top-k from the count-init table and implement it. I should answer a couple of questions:

Should we go for the 11L model? Probably yes, but only after addressing the bigram head or speeding things up.
EMA could be a good choice, especially if it’s implemented quickly. It might be better than SWA, which is similar but takes more time and lines of code. If there's a strict line budget and a decreasing loss trend, it’s best to replace SWA with EMA—definitely a key insight!Evaluating code complexity and optimization strategies
Considering additional code complexity, it might make sense to simplify by sticking with just the Exponential Moving Average (EMA) instead of stacking with Stochastic Weight Averaging (SWA) as the user mentioned from previous requests. It's a good suggestion since EMA could provide similar improvements with less overhead.

Next, about the bigram head: it likely isn't helpful since it's redundant with the BigramHash, making the exact dense version not worth the extra steps. It might be better to replace it with a small sparse top-k or revert to BigramHash 2048.

For step time overhead, it includes the bigram head and XSA, while Layer Normalization scale multiplication is negligible. The count-init prepass does not affect per-step time, and I wonder if using fullgraph=True compile could handle dynamic shapes adequately. It seems like the exact head might be creating a (N,1024) dense tensor before addition, which could saturate High Bandwidth Memory (HBM).

Another thought: utilizing partial RoPE via torch.cat with each step and possibly replacing XSA's repeat_interleave normalization might help too. The count-init pre-pass isn't affecting per-step time, and the LN scale is very slight. I think it’s worth trying LeakyReLU(0.5)² since it’s a quick adjustment. Finally, about shared value embedding, that's a medium-risk proposition without more details.Considering optimizations and changes

I think using FlashAttention 3 should only happen if native in the current torch version; it’s not a priority right now unless it can be easily integrated.

For the changes I’d suggest, it would be wise to match the winning stack by removing the exact head, going for 11 layers, adding EMA only, implementing a smaller batch sweep, and possibly introducing a sparse bigram residual along with temperature calibration.

I should mention that the most likely way to close that 0.03 gap isn't just adding tricks but focusing on speed and matching proven recipes. It's clear that the current run appears undertrained, emphasizing that an extra 800 steps could be crucial. So, prioritizing throughput becomes key.

I'll draft a structure with priority rankings and actionable snippets, possibly adding detail tags since there are many sections. Utilizing markdown and math formatting with ... seems essential. While it may feel overly detailed, the user requested deep thinking, so I'll provide a structured level of detail.Providing code snippets for improvements

I think it’s helpful to include some code snippets for clarity.

For an efficient Exponential Moving Average (EMA) implementation, I can initialize it like this:


# init
ema_decay = 0.997
ema_state = {k: v.detach().clone() for k, v in base_model.state_dict().items() if v.is_floating_point()}
# after every optimizer step
with torch.no_grad():
  for k, v in base_model.state_dict().items():
      if k in ema_state:
         ema_state[k].lerp_(v.detach(), 1.0 - ema_decay)
However, since state_dict returns new tensors, it might be more efficient to use named_parameters and named_buffers. I can maintain a parameter list with:


ema_params = [(n, p) for n, p in base_model.named_parameters()]
Updating with a foreach method could be simpler. There’s also the consideration that with DDP, updates should remain identical across ranks, but that should be manageable.

For temperature calibration, I can set it up like this:


@torch.no_grad()
def fit_temperature(model,...):
    tau = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([tau], max_iter=25, line_search_fn="strong_wolfe")
    ... T = tau.exp()
    loss = F.cross_entropy((logits/T).reshape(-1, V), y.reshape(-1), reduction="mean")
Then, during evaluation, simply use:


logits = logits / self.final_temp
```**Implementing a sparse top-k bigram residual**

I'm thinking about building a sparse top-k bigram residual. This can start from either the initialization or from a trained dense table. Here’s an outline of how to do that with a class:
```python
class SparseBigramBias(nn.Module):
    def __init__(self, vocab_size, k=16):
         self.top_idx = nn.Parameter(torch.zeros(vocab_size, k, dtype=torch.int16), requires_grad=False) 
         self.top_val = nn.Parameter(torch.zeros(vocab_size, k, dtype=torch.float32))
         self.scale = nn.Parameter(torch.tensor(1.0))
     
    def forward_add(self, logits_proj, prev):
         idx = self.top_idx[prev]          
         val = self.top_val[prev] * self.scale
         return logits_proj.scatter_add(1, idx.long(), val.to(logits_proj.dtype))
However, storing the int16 parameter might not be ideal. I could instead register it as a buffer, while perhaps making top_val trainable but fixing the indices from counts. This approach can be quite novel and effective. If time is tight, I could build from the count-init table and freeze it entirely since it acts like a tiny count-LM residual. Also, I need to remember to consider replacing the exact bigram head with either a bigram hash or a sparse head. Lastly, I should mention the batch size sweep and the environment variables used in the code!Considering efficiency enhancements

I don't need to provide code for EMA and replacing SWA, but that's an option to consider. For speeding up XSA, I might use the following snippet:


if self.use_xsa:
    vn = v * torch.rsqrt((v*v).sum(dim=-1, keepdim=True) + 1e-6)
    if self.num_kv_heads != self.num_heads:
        vn = vn[:, :, None].expand(-1, -1, self.num_heads // self.num_kv_heads, -1, -1).reshape(bsz, self.num_heads, seqlen, self.head_dim)
    y = y - (y * vn).sum(dim=-1, keepdim=True) * vn
This approach might avoid copying since .expand() doesn't allocate new memory until reshaped. I could consider using XSA only on the last two layers to save computation if all four aren’t necessary. Additionally, I could optimize the layer normalization scaling by precomputing constants as floats or buffers to skip the sqrt on each step. I'm pondering sharing value embeddings in later layers to reduce parameters and computation—it’s medium-risk, though. Lastly, I wonder if 8% magnitude pruning is too aggressive.Examining pruning strategies

I’m considering whether recent pruning changes may help or actually hurt the model's quality. With a 1.13MB headroom, reducing the pruning to 3-5% seems wise. The user increased pruning from 3% to 8%, which might be excessive since some headroom remains. If the extra bytes improve compression, that’s one thing, but if not, I’d suggest reducing pruning, perhaps back to 3% or even 0%, especially with the exact head removed. This could help recover some quality. Also, the bigram table in int4 may not be necessary if we’re dropping the exact head. We could allocate those bytes for an 11L model or a larger bigram hash.Assessing training efficiency

It seems the biggest issue isn't really the artifact size anymore, but rather training efficiency. My suggestion would be to focus on optimizing steps instead of pursuing clever compression methods. Here’s a suggested ranking for expected gains:

Remove the exact dense bigram head and swap it for a sparse top-k or BigramHash2048, which could offer a gain of +0.01 to +0.02 if the overhead is indeed 4ms.
Moving to 11L after a speed-up could provide an additional +0.006 to +0.012.
Replacing or adding EMA to SWA could bring in +0.003 to +0.006.
Implementing LeakyReLU squared might yield +0.001 to +0.003.
Temperature calibration could add +0.001 to +0.002.
Finally, a batch-token sweep could net +0.003 to +0.01, depending on batch size.
I should also clarify that the count-init prepass likely doesn't factor into the training wallclock. The submission evaluation includes training time. The overall script should fit within the 10-minute limit.Clarifying training and compression concerns

It looks like the prepass before training probably counts, but since it's just a one-time event lasting a few seconds, it shouldn’t affect training per step. If the max wallclock for the script is a consideration, the prepass might matter more. The competition likely evaluates end-to-end, meaning wallclock includes everything until the final model.

I think they aim for training and evaluation, so I'll suggest keeping the prepass under 2-3 seconds. Also, if budgeting is tight, I might recommend replacing SWA with EMA, since EMA improves performance without needing heuristics. I’ll also need to address the concerns about zstd compression with the bigram table and how it may worsen with overpruning.Considering layer addition and training implications

The user has 1.13MB of headroom, which means they could potentially add an 11th layer and reduce pruning. However, adding that layer will increase the compute needed and lower the steps, so it should probably only be done after any speed improvements. I think I could suggest lowering the sequence length if it doesn't harm performance, especially since 2048 may not be the best fit anyway. If they’re using 8 GPUs, the grad accumulation steps can remain fixed at 1. Lastly, I’ll mention the additional overhead from using a BigramLogitHead per token, as it could significantly increase elements during training.Explaining overhead and optimizing with sparse bigram

I see that the 4ms overhead can be clarified here. The count-init advantage mostly influences the first few hundred steps, but after around 6,000 steps, the model can still learn a small bigram hash effectively. Given a compute cap, I think it makes sense to focus on simpler inductive bias approaches. I should produce a code snippet for a sparse top-k bigram to show how we can count bigrams, pick the top k by residual magnitude, and then set up a fixed indices buffer with trainable values from count residuals. The values will be trainable, while the indices remain fixed, making this a more efficient setup.Building sparse bigram residual

To create a sparse bigram residual from counts, I have a function that starts by initializing bigram residuals. It uses torch.topk to find the top k values based on their absolute magnitudes, while also ensuring that the indices remain contiguous. I realize I need to gather the original signed values after getting the indices.

A good idea is to freeze the indices and train only the values to maintain simplicity. For a cheaper alternative with an exact head-like effect, I might suggest using a low-rank residual conditioned on the previous token. However, sparse top-k remains a more cost-effective approach. Additionally, there's potential in exploring a "Shared Value Embedding" that relates to shared low-rank projections, but it's important to focus on major improvements first.Exploring optimizations with XSA

I'm considering the possibility of combining with XSA. It utilizes a normalized v, and sharing value embeddings in the last layers might help reduce parameters and computational load, though it’s not essential.

Regarding "FlashAttention 3", since the user is using PyTorch SDP with the Flash backend, the built-in kernel might already perform similarly to FA2, making an external FA3 import a lower priority unless it's integrated into the installed Torch version. The benefits may not surpass the enhancements from 11L, EMA, or throughput fixes.

I'd suggest hard-coding all booleans and dimensions for compile-time DCE (dead code elimination) efficiency and separating attention classes for XSA and standard versions to avoid runtime checks. This separation could simplify the graph and speed things up, especially with the QAT branch facing DCE issues. If the last four layers use XSA, I recommend creating different module classes rather than using conditionals within the forward method. Also, for rope_dims, consistency might be beneficial.Optimizing code and performance

I’m thinking that separating the last four layers into an XSABlock class could help eliminate branches and save on compile overhead, although the impact might be minor. However, there’s a line limit to consider, so I need to be cautious about adding too much extra code.

For my response, I’ll include a blunt diagnosis, suggestions for next steps, and code snippets, along with answers to specific questions.

Here’s the outline:

Blunt Diagnosis:

It looks like you're underutilizing bytes while overusing compute on the dense bigram head.
The gap to 1.1248 seems to be from throughput and recipe mismatch, not quantization.
With 1.13MB headroom and 8% pruning, you're byte-rich but step-poor.
Direct Answers:

11L: Yes, but only after cutting the dense bigram head or improving the speed path, otherwise you might lose too many steps.
EMA: Yes, I’d suggest replacing SWA or at least evaluating EMA-only first.
BigramLogitHead: Likely not worth using the dense version.Formulating an optimization plan
I'm considering replacing the current structure with either BigramHash(2048) or a sparse top-k residual. The main overhead in step time seems to come from the dense bigram head and XSA, while LeakyReLU² is a yes.

For the ranked plan to close the performance gap:
A. Start with a throughput-first run: switch to 10L, remove the exact head, use BigramHash2048, LeakyReLU², EMA-only, and apply 0-3% pruning. Target steps between 6800-7100.
B. If speed improves, transition to 11L while regaining some bytes.
C. Add temperature calibration.
D. Consider using a sparse top-k bigram residual for novelty instead of the dense head.

I’ll also prepare code snippets for the changes mentioned, including adjustments for LeakyReLU², EMA, temperature calibration, and the sparse bigram residual class initialization. Additionally, I might touch on batch size environment sweep.Estimating performance improvements

To optimize the model, the first step is analyzing current overheads: I estimate 4ms from the dense bigram head and 3ms from XSA, which brings performance down to around 92.4 ms. If the top PR shows 85ms, there are likely other dependencies, including potential implementation efficiencies.

To address this, I recommend dropping the dense bigram head for either BigramHash(2048) or a sparse top-k residual. Let's prioritize a few actionable steps:

Transition to 11L and consider reducing the logit soft cap if using LeakyReLU².
Replace SWA with EMA, targeting about 0.997 decay.
Incorporate temperature calibration.
Suggest pruning only 0-3%.
Implement code snippets as outlined, which include necessary changes for LeakyReLU² and temperature adjustments.
Lastly, the frozen count-LM interpolation could be a clever strategy to minimize training overhead while providing improved evaluation performance.Calculating performance gains

To enhance performance, I'm considering multiplying by 11/10, which results in a +4.5% improvement compared to the 10L with m=3 configuration. That's exciting! I should include this adjustment—suggesting an "11L, mlp_mult=2.75" snippet in the final answer.

I want to make sure my response is concise yet meaningful, so I'll compile these insights clearly. This structured presentation will help communicate the advantages of the proposed changes effectively.
Yes — I think you still have **real room left**, but the main issue is now **not compression tricks**. It’s **throughput vs bytes**.

## My blunt read

You are currently:

- **byte-rich**: artifact is only **14.81 MB**, so you’re leaving about **1.19 MB** on the table
- **step-poor**: your run hits the wallclock while the curve is still dropping fast

That means the next gains probably come from:

1. **using the extra bytes better**
2. **getting more useful optimization steps**
3. **removing expensive features that don’t clearly beat the winning stack**

---

# What I would change first

## Priority 1: Stop paying 4 ms/step for a dense exact bigram head

I would **not** keep the dense `BigramLogitHead` as the mainline.

Why:

- it adds memory traffic every step
- the top PRs do **not** need it
- your own results don’t show a decisive win
- you are losing steps under a hard 10-minute cap

### My recommendation
Replace it with one of these:

- **`BigramHash(2048)`** like the top PRs
- or a **sparse top-$k$ bigram residual** from counts

If you want the “count-init” advantage without dense cost, **sparse top-$k$ residual** is the best compromise.

---

## Priority 2: Use your spare bytes for architecture, not pruning

You have:

- **1.13 MB headroom**
- **8% pruning**

That combination is a red flag.

If you already fit comfortably, **8% pruning is probably costing quality for no reason**.

### I would do one of these
- go to **11 layers**
- reduce pruning back to **0% to 3%**
- or both

If you keep 10L and 8% pruning while sitting at 14.8 MB, you are probably underusing the budget.

---

## Priority 3: EMA is probably worth more than another fancy feature

I would add **EMA**, but I would seriously consider **EMA instead of SWA first**, not necessarily both.

Reason:

- EMA is simple
- it helps when training is still moving at the end
- your curve suggests you are **not fully converged**
- SWA is better when you’re oscillating around a basin; EMA is often better when you’re still descending

So my first test would be:

- **EMA(0.997)** only
- compare to SWA-only
- only keep both if EMA-only clearly underperforms

---

## Priority 4: Try **11L with slightly smaller MLP** to preserve speed

This is the most important structural suggestion.

Instead of jumping from:

- 10L, `mlp_mult=3.0`

to

- 11L, `mlp_mult=3.0`

try:

- **11L, `mlp_mult=2.7` to `2.8`**

Why this is smart:

- depth helps a lot at this scale
- MLP dominates a big chunk of block compute
- slightly shrinking MLP can offset much of the extra layer cost

Very rough compute intuition:

- block cost is roughly attention + MLP
- with `mlp_mult = 3.0`, MLP is a large share
- changing $m: 3.0 \to 2.75$ cuts MLP cost by about $8\%$
- adding an 11th layer adds about $10\%$ depth

So **11L with a slightly smaller MLP** can be a very good compute-neutral trade.

This is one of the best “contest-specific” moves available.

---

# Direct answers to your questions

## 1. Should you go 11L?
**Yes — but not with the current dense bigram head.**

Best order:

1. remove dense exact bigram head
2. reduce pruning
3. then try **11L**
4. if step time gets too high, use **`mlp_mult=2.7` or `2.8`**

If you keep the expensive head **and** add a layer, you may lose too many steps.

---

## 2. Should you add EMA?
**Yes.**

But I would test in this order:

1. **EMA only**
2. **SWA only**
3. EMA + SWA only if clearly better

If line budget is tight, EMA-only is the cleaner bet.

---

## 3. Is the exact bigram head helping?
My guess: **probably not enough to justify its cost**.

The strongest evidence is not theory — it’s your setup:

- top PRs use **BigramHash**
- your dense head costs steps
- your run is still learning hard at the cap

Under a fixed wallclock, a clever feature that costs updates can lose even if it improves the model slightly per step.

### My verdict
- **dense exact head**: probably cut
- **sparse count residual**: worth trying
- **BigramHash(2048)**: safe fallback

---

## 4. What is really causing 96 ms vs 85 ms?
### Not these:
- **LN Scale multiply**: basically negligible
- **count-init pre-pass**: startup-only, not per-step

### Likely causes:
- dense **BigramLogitHead**
- some overhead in your XSA implementation
- maybe a bit from extra tensor reshapes / `torch.cat`
- maybe not exactly apples-to-apples step timing vs PRs

The dense bigram head is the most obvious culprit.

---

## 5. LeakyReLU$(0.5)^2$ vs ReLU$^2$?
**Yes, try it.**  
It’s almost free and should absolutely be in your sweep.

---

## 6. Shared Value Embedding / FlashAttention 3?
### Shared Value Embedding
Maybe interesting, but **not my next experiment** unless you can copy a known-good implementation cheaply.

### FlashAttention 3
Only if it is effectively **free via your installed PyTorch stack**.

If it requires extra dependencies or fragile code paths, I would **not** chase it before:
- 11L
- EMA
- batch sweep
- dropping the dense bigram head

---

## 7. What would I change to close the gap?
In order:

1. **Drop dense exact bigram head**
2. **Reduce pruning from 8% to 0–3%**
3. **Add EMA**
4. **Try 11L**
5. **Try 11L with `mlp_mult=2.75`**
6. **Switch to LeakyReLU$(0.5)^2$**
7. **Do a batch-size sweep**
8. **Add post-quant temperature calibration**
9. If you still want a local-feature trick: **sparse top-$k$ bigram residual**

---

# Biggest missing idea: sparse bigram residual

This is the cleanest “clever hack” I think you’re still missing.

Instead of a dense $1024 \times 1024$ table, keep only the strongest $k$ next-token biases per previous token.

## Why it’s good
- preserves most of the count-LM benefit
- tiny artifact cost
- much lower compute than dense bigram logits
- easy to initialize from your count table

For example with $k=16$:

- pairs stored: $1024 \cdot 16 = 16384$
- indices + values are tiny
- compute is $O(Nk)$ scatter-add, not full dense table gather

---

## Sparse bigram residual code sketch

### Build from counts
```python
def build_sparse_bigram_residual(
    pattern: str,
    vocab_size: int,
    sample_tokens: int = 16_000_000,
    alpha: float = 0.25,
    clip_value: float = 4.0,
    k: int = 16,
):
    B = build_bigram_residual_init(
        pattern=pattern,
        vocab_size=vocab_size,
        sample_tokens=sample_tokens,
        alpha=alpha,
        clip_value=clip_value,
    )  # [V, V]

    idx = B.abs().topk(k, dim=1).indices  # [V, k]
    val = B.gather(1, idx)                # signed values [V, k]
    return idx.to(torch.int16).contiguous(), val.to(torch.float32).contiguous()
```

### Module
```python
class SparseBigramResidual(nn.Module):
    def __init__(self, vocab_size: int, k: int, idx_init: Tensor, val_init: Tensor):
        super().__init__()
        self.register_buffer("idx", idx_init, persistent=True)   # [V, k], int16
        self.val = nn.Parameter(val_init)                        # [V, k], float/bf16
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def add_to_logits(self, logits_proj: Tensor, prev_tokens: Tensor) -> Tensor:
        # logits_proj: [N, V]
        idx = self.idx[prev_tokens].long()  # [N, k]
        val = self.val[prev_tokens].to(dtype=logits_proj.dtype)
        return logits_proj.scatter_add(1, idx, self.scale.to(logits_proj.dtype) * val)
```

### Use before softcap
```python
if self.sparse_bigram is not None:
    prev = input_ids.reshape(-1)
    logits_proj = self.sparse_bigram.add_to_logits(logits_proj, prev)

logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
```

This is the exact replacement I would test if you want to keep the “count-based local model” idea without dense overhead.

---

# EMA code snippet

If you want minimal EMA:

```python
ema_decay = 0.997
ema_state = {
    k: v.detach().clone()
    for k, v in base_model.state_dict().items()
    if v.is_floating_point()
}

# after optimizer.step()
with torch.no_grad():
    cur = base_model.state_dict()
    for k, v in cur.items():
        if k in ema_state:
            ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
```

At the end:

```python
def load_ema_into_model(model: nn.Module, ema_state: dict[str, Tensor]) -> None:
    sd = model.state_dict()
    for k, v in ema_state.items():
        sd[k].copy_(v.to(dtype=sd[k].dtype, device=sd[k].device))
```

If line budget is tight, EMA can replace SWA.

---

# LeakyReLU$^2$ code

Very easy:

```python
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())
```

This should definitely be in the next sweep.

---

# Post-quant temperature calibration

This is low-cost and probably worth it.

## Why
Quantization often changes confidence more than ranking. A scalar temperature can recover a bit of CE/bpb.

### Minimal fitter
```python
def fit_temperature(model: nn.Module, x_cal: Tensor, y_cal: Tensor, device: torch.device) -> float:
    tau = torch.zeros((), device=device, requires_grad=True)  # T = exp(tau)
    opt = torch.optim.LBFGS([tau], lr=0.1, max_iter=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model.forward_logits(x_cal).reshape(-1, model.tok_emb.num_embeddings).float()
            T = tau.exp().clamp_min(1e-4)
            loss = F.cross_entropy(logits / T, y_cal.reshape(-1), reduction="mean")
            loss.backward()
            return loss

    opt.step(closure)
    return float(tau.exp().item())
```

Then in eval:
```python
logits = logits / final_temperature
```

This is the kind of tiny final-stage polish that matters in a contest.

---

# Batch-size sweep: I think this is underexplored

Because you are **wallclock-limited**, not epoch-limited, I would absolutely sweep:

- `524288`
- `655360`
- `786432`

for `TRAIN_BATCH_TOKENS`.

If the curve is still steep at the end, a smaller batch can win by giving you more updates.

This is one of the most contest-relevant knobs and costs almost no code.

---

# My recommended next 4 runs

## Run A — safest path to match winning recipe
- remove dense bigram head
- `BigramHash(2048)`
- 10L
- EMA
- LeakyReLU$(0.5)^2$
- pruning back to `0.03` or `0.00`

## Run B — spend bytes better
- same as Run A
- **11L**
- same MLP 3.0 if step time stays okay

## Run C — depth-for-MLP trade
- **11L**
- `mlp_mult = 2.75`
- EMA
- LeakyReLU$(0.5)^2$

## Run D — your novel local-feature variant
- remove dense bigram head
- use **sparse top-$k$ count residual** with $k=8$ or $16$
- compare against BigramHash(2048)

---

# Final recommendation

If you want the shortest path upward, I would do this:

## Best next change set
1. **Drop dense exact bigram head**
2. **Reduce pruning from 8% to 0–3%**
3. **Add EMA**
4. **Switch to LeakyReLU$(0.5)^2$**
5. **Try 11L**
6. If 11L is too slow, use **11L + `mlp_mult=2.75`**

And if you want one clever extra idea beyond the top PR stack:

7. **replace dense bigram head with sparse top-$k$ count residual**

That is the most plausible way to get the “count-LM bonus” **without paying dense-logit cost**.

If you want, I can turn this into a **patch list** against your current `train_gpt_submission.py` with the smallest code edits first.