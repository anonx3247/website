# Criticality-Inducing Training for LLM Swarms — v1 Design Document

**Author**: Anas (research notes, draft)
**Status**: v1 design, pre-implementation
**Infra**: Tinker (Thinking Machines Lab)
**Scope**: Toy task (multi-fragment NIAH), GRPO + LoRA, swarm of 64 agents

---

## 1. Motivation

Hierarchical multi-agent LLM systems (orchestrator + subagents, e.g. Kimi K2.5's PARL) achieve good task performance but inherit the brittleness of their hierarchy: the orchestrator is a single point of failure, scaling is bottlenecked, and the system has hand-engineered communication topology. A leaderless alternative — Boids-style swarm coordination with emergent communication patterns — is genuinely under-explored for LLMs.

For a leaderless swarm to be computationally useful, it must operate near a **critical regime**: the phase boundary between local independence (no coordination) and global lockstep (mode collapse). At criticality, information propagates across the swarm at all scales, susceptibility to relevant inputs is maximized, and dynamic range is largest. Off-critical, either coordination dies out or the swarm collapses to consensus on noise.

This document specifies v1 of an experimental program to test whether **criticality-inducing objectives** can be added to RL training of an LLM swarm such that emergent dynamics become critical, measured by power-law avalanche statistics with consistent finite-size scaling exponents, and whether such criticality is computationally productive on a controlled task (multi-fragment needle-in-a-haystack).

---

## 2. Task: Multi-Fragment NIAH

### 2.1 Definition

A **rollout** consists of:
- A **corpus** $C$: long irrelevant text (Paul Graham essays for v0, RULER/BABILong-derived for v1) of total length $L_C$ tokens, partitioned into $N$ shards $\{C_1, \dots, C_N\}$ of roughly equal length.
- A **question** $q$ whose answer is a structured object $a^* = (s_1^*, \dots, s_K^*)$ with $K$ slots (e.g. $K=3$: name, function, event).
- $K$ planted **fragments** $f_1, \dots, f_K$, each a short text passage that uniquely encodes one slot value, inserted at controlled positions across shards. Fragments are placed such that **no shard contains all $K$ fragments** — by construction, every solution requires at least 2 agents' contributions.
- **Decoy fragments**: passages that fill some slots correctly but conflict on others (correct function and event, wrong name). These force cross-checking rather than first-match commitment.

### 2.2 Difficulty axes

| Axis | Symbol | v1 default | Range |
|---|---|---|---|
| Number of agents | $N$ | 64 | $\{32, 64, 128, 256\}$ |
| Number of slots | $K$ | 3 | $\{2, 3\}$ |
| Corpus length | $L_C$ | 5M tokens | $\{1M, 5M, 20M\}$ |
| Per-agent shard | $L_C/N$ | ~78K @ default | (sized to fit context) |
| Decoys per slot | $D$ | 2 | $\{0, 1, 2, 4\}$ |
| Search budget | $B$ | $0.3 \cdot L_C$ | tighter $\Rightarrow$ harder |

Corpus is much larger than any single agent's context window. The search budget $B$ is the total number of corpus tokens the swarm may collectively read. With $B < L_C$, brute-force coverage is impossible; coordination becomes about *efficient* allocation rather than mere completeness.

The task is structured so that **success is achievable but efficiency is the discriminator**. The gradient signal lives in coordination quality, not in task feasibility.

### 2.3 Scoring

For a reported answer $\hat{a} = (\hat{s}_1, \dots, \hat{s}_K)$:

$$R_{\text{task}}(\hat{a}, a^*) = \beta \cdot \mathbb{1}\!\left[\bigwedge_{k=1}^K \text{match}_k(\hat{s}_k, s^*_k)\right] + \sum_{k=1}^K \alpha_k \cdot \mathbb{1}\!\left[\text{match}_k(\hat{s}_k, s^*_k)\right] - \gamma \cdot \frac{B_{\text{used}}}{B}$$

- $\text{match}_k$: canonical equality for structured slots (names, functions, dates), calibrated cosine-similarity threshold for free-text slots (events).
- $\beta$: completion bonus (discontinuity at full resolution).
- $\alpha_k$: per-slot partial credit.
- $\gamma$: budget-utilization penalty (rewards efficiency).

---

## 3. Mathematical Formalism

### 3.1 Swarm and policy

A swarm of $N$ agents shares a single LoRA-adapted policy $\pi_\theta$. Base model: small open-source LLM (Qwen-2.5 7B target, Qwen-2.5 3B for fast iteration). LoRA: rank 16, $\alpha=32$, applied to attention projections and MLP up/down. Base frozen.

Agent $i$ at step $t$ has local state $x_i^t = (c_i, m_i^t, h_i^t)$:
- $c_i$: corpus shard (static within rollout)
- $m_i^t$: messages received from communication-graph neighbors up to $t$
- $h_i^t$: agent's CoT/scratchpad accumulated so far

### 3.2 Communication topology (v1: random k-regular graph)

For each rollout, sample a random $k$-regular graph $\mathcal{G}$ on $N$ vertices ($k=4$ default). An agent's broadcasts are visible only to its $k$ graph-neighbors. Graph fixed within rollout, resampled across rollouts.

**v1 simplification**: static random topology. v2 will replace with dynamic semantic locality (sentence-transformer embeddings of CoTs determine neighborhoods). v1 establishes whether criticality dynamics work under simple structural assumptions.

### 3.3 Action space

Agents are LLMs producing CoT and tool calls. We do not introduce a separate discrete action space; tool calls *are* the actions:

| Tool call | Action | Effect |
|---|---|---|
| `<read span="...">` | READ | consumes $|\text{span}|$ from budget; reveals span |
| `<broadcast msg="...">` | BROADCAST | emits message to graph-neighbors, gated by $g_i^t$ |
| `<report answer="...">` | REPORT | submits final answer; ends rollout |
| (any other text — CoT) | NOOP | no environmental effect |

The broadcast gate $g_i^t \in [0,1]$ is produced alongside the tool call — the policy emits both message and confidence/gate score. Broadcast realized iff $g_i^t > \tau$ ($\tau=0.5$ fixed in v1). This is the SOC threshold mechanism.

### 3.4 Block-level credit assignment

A trajectory is segmented into **blocks** by tool-call boundaries:

```
[CoT_1] <read>  [CoT_2] <broadcast>  [CoT_3] <report>
 NOOP    READ    NOOP     BROADCAST   NOOP    REPORT
```

Advantages are computed at block granularity. Within a block, all tokens share the block's advantage. Block contribution to loss is normalized by length so a 200-token CoT doesn't dominate a 5-token tool call:

$$\mathcal{L}^{\text{block}} = \frac{1}{|\text{block}|} \sum_{t \in \text{block}} \rho_t \hat{A}_{\text{block}}$$

This keeps GRPO standard while matching credit structure to the task's natural granularity.

### 3.5 Causal-broadcast DAG

A broadcast $B_i$ at time $t_i$ **causally influences** broadcast $B_j$ at $t_j > t_i$ if:

(a) $B_i$ was received by agent $j$ before $t_j$, AND
(b) the counterfactual gate logit of agent $j$ — recomputed without $B_i$ in $j$'s message buffer — falls below $\tau$, while the actual gate logit is above $\tau$.

The receipt of $B_i$ tipped $j$ into broadcasting. We build a causal DAG over all broadcasts in a rollout using these edges.

**Computational cost**: counterfactual gate logits require an extra forward pass per (broadcast, prior-broadcast-in-buffer) pair. In v1 we subsample to 25% of pairs during training; full computation at evaluation only.

### 3.6 Productive cascades and slot-fill leaves

A `<report>` action that contains a correctly matched slot value is a **slot-fill event** $e$.

**Leaf identification (v1, cheap)**: Broadcast $B$ is a leaf for slot-fill event $e$ if (a) $B$ was received by the reporting agent before the report, (b) $B$ was emitted within a recency window $W$ before the report (default: last 4 swarm-steps), and (c) the report's CoT contains a textual or semantic reference to content in $B$. Reference is checked via substring match on key tokens plus cosine-similarity of $B$'s message against report-CoT segments above threshold $0.7$.

**Leaf identification (alternative, principled)**: counterfactually verify that without $B$ in the reporting agent's buffer, the agent would not have produced the same correct answer. Consistent with the causal-edge mechanism in §3.5 but adds one forward pass per candidate leaf. Reserved for v1.5 / final eval.

A **productive cascade** is a weakly-connected component in the causal DAG that contains at least one leaf broadcast. Non-productive causal cascades are logged for diagnostics but not used in the loss.

### 3.7 Attribution weighting (recursive DAG-propagation)

For each slot-fill event $e$ with reward $R_e$, we distribute reward backward through the causal DAG using uniform per-parent shares.

Define attribution $A(B)$ recursively:

$$A(B) = \sum_{B' \,:\, B \to B' \text{ in DAG}} \frac{A(B')}{|\text{parents}(B')|}$$

with base case $A(B_{\text{leaf}}) = R_e$ for any broadcast that is a leaf for slot-fill $e$.

For multiple slot-fills in one rollout, sum across events: $A^{\text{total}}(B) = \sum_e A_e(B)$.

**Worked example**.
```
B1, B2 → B3
B4, B5, B6 → B7
B3, B7 → B8 → answer
```
With reward $R$ at B8 (the leaf):
- $A(B_8) = R$
- $A(B_3) = R / |\text{parents}(B_8)| = R/2$
- $A(B_7) = R/2$
- $A(B_1) = (R/2)/|\text{parents}(B_3)| = R/4$
- $A(B_2) = R/4$
- $A(B_4) = (R/2)/|\text{parents}(B_7)| = R/6$
- $A(B_5) = R/6$
- $A(B_6) = R/6$

This scheme gives:
- **Depth discounting**: deep upstream broadcasts get less credit than proximal ones.
- **Width discounting**: a broadcast whose effect was diluted across many siblings gets a smaller share.
- **Diamond handling**: a broadcast contributing through multiple downstream paths gets credit through each, summed.
- **Compositional across slot-fills**: total attribution is the sum across all slot-fill events.

Per-agent attribution: $A_i = \sum_{B \text{ emitted by } i} A^{\text{total}}(B)$.

### 3.8 Critical exponents (target)

If the trained swarm is critical:

$$P(s) \sim s^{-\tau}, \quad P(T) \sim T^{-\alpha}, \quad \langle s \rangle(T) \sim T^\gamma$$

with self-consistency $\gamma = (\alpha-1)/(\tau-1)$. Mean-field branching predicts $\tau = 3/2, \alpha = 2, \gamma = 2$. We don't expect mean-field exactly; we do expect (a) approximate power laws over $\geq 1.5$ decades, (b) self-consistent exponents, (c) data collapse under finite-size scaling across $N \in \{32, 64, 128, 256\}$.

Statistics computed over **productive cascades**; the chatty/non-productive distribution is logged separately as a control.

---

## 4. Loss Function

### 4.1 GRPO formulation (nested groups)

Tinker's `EnvGroupBuilder` natively supports two group semantics, which we use simultaneously:

- **Outer group (across rollouts)**: $G$ rollouts of the *same task config*. Standard GRPO variance reduction — advantages centered against the outer-group mean.
- **Inner group (within rollout)**: the $N$ agents in a single rollout. Each agent's trajectory is a sample of the shared policy in a different state. Per-agent attribution (§3.7) gives heterogeneous credit; centering across agents within a rollout gives the swarm-level credit signal.

For agent $i$ in rollout $g$ with rollout-level reward $R_g$ and attribution $A_i^g$:

$$R_i^g = R_g \cdot w_i^g, \qquad w_i^g = \sigma\!\left(\lambda_w \cdot \frac{A_i^g}{\max_j A_j^g + \epsilon}\right)$$

The participation weight $w_i^g$ rewards agents who appear at high-credit positions in productive cascades (proximal, narrow-fan-in). $\lambda_w = 2.0$ initial.

**Two-level advantage**. We center first within rollout, then across rollouts:

$$\hat{A}_i^g = \frac{R_i^g - \mu_g^{\text{inner}}}{\sigma_g^{\text{inner}} + \epsilon} + \zeta \cdot \frac{R_g - \mu^{\text{outer}}}{\sigma^{\text{outer}} + \epsilon}$$

where $\mu_g^{\text{inner}} = \text{mean}_j(R_j^g)$, etc., and $\zeta \in [0,1]$ blends outer-group signal (default $\zeta = 0.5$). Inner term gives credit assignment among agents; outer term reduces variance against task difficulty.

GRPO loss with PPO clip:

$$\mathcal{L}_{\text{GRPO}}^{\text{swarm}} = -\mathbb{E}_{g, i}\!\left[\frac{1}{|\tau_i^g|} \sum_{\text{block} \in \tau_i^g} \frac{1}{|\text{block}|}\sum_{t \in \text{block}} \min\!\big(\rho_t \hat{A}_i^g,\; \text{clip}(\rho_t, 1{-}\epsilon, 1{+}\epsilon) \hat{A}_i^g\big)\right] + \beta_{\text{KL}}\, \mathcal{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$$

with $\rho_t = \pi_\theta(a^t \mid x^t) / \pi_{\theta_{\text{old}}}(a^t \mid x^t)$.

The shared LoRA receives gradients from all $N \times G$ agent-trajectories pooled. Agent interchangeability preserved: identical policy, heterogeneous states.

### 4.2 Mutual information term (attribution-weighted)

Let $z_B \in \mathbb{R}^d$ be the residual-stream activation of the policy at the moment broadcast $B$ is emitted (final hidden state at the BROADCAST tool token). For each causal edge $B_p \to B_c$ in a productive cascade, compute InfoNCE:

$$\mathcal{L}_{\text{MI}} = -\mathbb{E}_{(B_p \to B_c)}\!\left[\omega_{p,c} \cdot \log \frac{\exp(\phi(z_{B_p}, z_{B_c})/T)}{\sum_{B_k \in \mathcal{N}^-} \exp(\phi(z_{B_p}, z_{B_k})/T)}\right]$$

- $\mathcal{N}^-$: negatives from non-causally-linked broadcasts in the same outer group.
- $\phi(u,v) = u^\top W v$: small bilinear critic, trained jointly.
- $T = 0.1$ initial.
- $\omega_{p,c} = \frac{A(B_c)/|\text{parents}(B_c)|}{\sum_{B} A^{\text{total}}(B)}$: edge-attribution weight — the share of reward flowing along this specific edge, normalized.

Load-bearing edges (high attribution flow) contribute more to the MI objective; peripheral edges contribute less. The restriction to productive-cascade edges prevents the failure mode where agents satisfy MI by broadcasting correlated noise — such broadcasts wouldn't be on causal paths to slot-fills, so their edges are never in the sum.

### 4.3 Communication sparsity

$L_1$ on broadcast gate values (continuous, gives smooth gradients):

$$\mathcal{L}_{\text{sparse}} = \frac{1}{N \cdot T_{\max}}\sum_{i=1}^N \sum_{t=1}^{T_{\max}} g_i^t$$

Combined with productive-restricted MI: silence is cheap, broadcasting expensive, productive broadcasting the only thing that pays.

### 4.4 Total loss

$$\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GRPO}}^{\text{swarm}} + \lambda_{\text{MI}} \mathcal{L}_{\text{MI}} + \lambda_{\text{sp}} \mathcal{L}_{\text{sparse}}}$$

**No FSS regularizer.** FSS is reserved as measurement (§6) — using it as a regularizer would assume the conclusion (criticality) rather than test for it.

---

## 5. Training Procedure (Tinker)

### 5.1 Mapping to Tinker primitives

| Concept | Tinker primitive |
|---|---|
| Single agent's view | `Env` (one per agent in swarm) |
| One swarm rollout | `EnvGroupBuilder` returning $N$ envs |
| Outer group (multiple rollouts of same task) | Multiple `EnvGroupBuilder` instances batched |
| LoRA training client | `service_client.create_lora_training_client(rank=16)` |
| Custom loss | `forward_backward_custom` |
| Inference for rollouts | `save_weights_and_get_sampling_client()` + `sample()` |
| Counterfactual gate logits | extra `sample()` calls with modified message buffer |

The InfoNCE bilinear critic $\phi$ is **outside Tinker** — a small CPU/GPU-local module (PyTorch, ~1M params) that we train alongside via standard backprop on logged residual activations. Gradients to the policy flow through Tinker's `forward_backward_custom`; gradients to $\phi$ are local.

### 5.2 Phases

Each phase ends when its criterion stabilizes for $\geq 100$ steps.

**Phase 0 — Task warmup (single agent).**
- $N=1$, full corpus visible (or ablated to fit context), no broadcast tool.
- Loss: $\mathcal{L}_{\text{GRPO}}$ only.
- Goal: agent uses READ + REPORT effectively.
- Criterion: $\geq 50\%$ task success on held-out questions at $L_C = 1M, K=2$.

**Phase 1 — Swarm task (no MI, no sparsity).**
- $N=64$, sharded, full action space.
- Loss: $\mathcal{L}_{\text{GRPO}}^{\text{swarm}}$ only ($\lambda_{\text{MI}} = \lambda_{\text{sp}} = 0$).
- Goal: swarm coordinates under task pressure (rollout reward + budget penalty).
- Criterion: $\geq 20\%$ improvement over no-comm baseline on efficiency-weighted success.
- Diagnostic: log productive-cascade statistics as pre-criticality baseline.

**Phase 2 — Criticality terms.**
- Same setup as Phase 1.
- Loss: full $\mathcal{L}_{\text{total}}$.
- Goal: productive avalanche distributions develop power-law shape; swarm performance improves or holds.
- Criterion: $P(s_{\text{productive}})$ log-log slope $\in [-3, -1]$ over $\geq 1$ decade with non-trivial p-value.

**Phase 3 — Multi-scale measurement.**
- Mixed batches, $N \in \{32, 64, 128, 256\}$.
- Loss: full $\mathcal{L}_{\text{total}}$ (no FSS regularization).
- Goal: critical exponents *measured* across $N$, self-consistency and data collapse verified.
- Criterion: scaling collapse residual $< 0.15$.

If Phase 3 fails to show data collapse, that's an interesting result on its own — locally power-law statistics without the universal scaling structure of a true critical point.

### 5.3 Training loop pseudocode (Tinker-flavored)

```python
# === Setup ===
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen2.5-7B-Instruct", rank=16, alpha=32
)
infonce_critic = BilinearCritic(d=hidden_dim).to(device)
optimizer_critic = AdamW(infonce_critic.parameters(), lr=1e-4)

for step in range(num_steps):
    # === Sample G outer-group rollouts of same task config ===
    cfg = sample_rollout_config()  # N, K, L_C, B, decoys, k
    sampling_client = training_client.save_weights_and_get_sampling_client()

    rollouts = []
    for g in range(G):
        # Each rollout is one EnvGroupBuilder with N agent-Envs
        rollout = run_swarm(sampling_client, cfg, graph=sample_k_regular(N, k=4))
        rollouts.append(rollout)

    # === Causal DAG + cascade analysis (offline, post-rollout) ===
    for r in rollouts:
        r.causal_DAG = build_causal_DAG(r, sampling_client)   # uses CF gate logits
        r.leaves = identify_leaves_cheap(r)                   # §3.6 v1 method
        r.productive_cascades = filter_productive(r.causal_DAG, r.leaves)
        r.attribution = propagate_attribution(r.causal_DAG, r.leaves, r.rewards)

    # === Two-level advantages ===
    R_outer = [task_reward(r) for r in rollouts]
    mu_o, sd_o = mean(R_outer), std(R_outer)
    for g, r in enumerate(rollouts):
        R_inner = [R_outer[g] * sigmoid(λ_w * r.attribution[i] / max_attr) for i in range(N)]
        mu_i, sd_i = mean(R_inner), std(R_inner)
        for i in range(N):
            r.A[i] = (R_inner[i] - mu_i) / (sd_i + ε) + ζ * (R_outer[g] - mu_o) / (sd_o + ε)

    # === Build Tinker training data ===
    training_data = []
    for r in rollouts:
        for i in range(N):
            for block in r.trajectories[i].blocks:
                training_data.append(Datum(
                    tokens=block.tokens,
                    advantage=r.A[i],
                    log_prob_old=block.log_prob,
                    block_length=len(block.tokens),
                ))

    # === GRPO loss via Tinker's forward_backward_custom ===
    grpo_loss_fn = make_clipped_surrogate_loss(epsilon=0.2, beta_kl=0.01)
    fb_future = training_client.forward_backward_custom(training_data, grpo_loss_fn)

    # === InfoNCE loss (separate, local) ===
    edges = [(p, c, ω) for r in rollouts
             for cascade in r.productive_cascades
             for (p, c, ω) in cascade.edges_with_attribution_weights]
    L_mi = infonce_loss(infonce_critic, edges, negatives=sample_negatives(rollouts))
    L_sp = sparsity_loss(rollouts)

    # The MI gradient flows to the policy through residual activations
    # (recorded during sampling, treated as policy outputs for the InfoNCE critic).
    # Implementation note: the InfoNCE term is added as a custom loss head
    # via forward_backward_custom that takes (policy_residuals, critic_output) → scalar.
    fb_future_aux = training_client.forward_backward_custom(
        residual_data, lambda r: λ_MI * L_mi(r) + λ_sp * L_sp(r)
    )

    fb_future.result()
    fb_future_aux.result()
    training_client.optim_step(...)
    optimizer_critic.step()  # local

    # === Logging ===
    log_avalanche_distributions(rollouts)
    log_productive_ratio(rollouts)
    if step % eval_interval == 0:
        evaluate_held_out(sampling_client)
        fit_critical_exponents(recent_productive_cascades)
        if phase >= 3: measure_finite_size_scaling()
```

### 5.4 Rollout (one swarm episode)

```python
async def run_swarm(sampling_client, cfg, graph):
    agents = [AgentEnv(shard=C_i, question=cfg.q) for i in range(cfg.N)]
    budget = cfg.B
    cascade_log = CascadeLogger()
    final_answer = None

    for t in range(cfg.T_max):
        if budget <= 0 or final_answer is not None:
            break

        # Step all agents concurrently (Tinker async sample)
        actions = await asyncio.gather(*[
            sample_block(sampling_client, agents[i], received=collect_msgs(graph[i], t))
            for i in range(cfg.N)
        ])

        for i, (block, action, gate, residual) in enumerate(actions):
            agents[i].cot.append(block)
            if action.type == READ:
                budget -= len(action.span)
                agents[i].observe(action.span)
            elif action.type == BROADCAST and gate > τ:
                broadcast(action.msg, sender=i, receivers=graph[i], t=t,
                          residual=residual, gate=gate,
                          received_at_emission=collect_msgs(graph[i], t))
                cascade_log.record(i, t, action.msg, residual, gate)
            elif action.type == REPORT:
                final_answer = action.answer
                break

    return Rollout(
        trajectories=[a.trajectory for a in agents],
        cascade_log=cascade_log,
        final_answer=final_answer,
        budget_used=cfg.B - budget,
    )
```

### 5.5 Hyperparameters

| Symbol | Meaning | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| $G$ | Outer group size | 4 | 4 | 4 |
| $N$ | Agents per rollout | 64 | 64 | varied |
| $\lambda_{\text{MI}}$ | MI weight | 0 | 0.1 | 0.1 |
| $\lambda_{\text{sp}}$ | Sparsity weight | 0 | 0.05 | 0.05 |
| $\lambda_w$ | Participation weight scale | 2.0 | 2.0 | 2.0 |
| $\zeta$ | Outer/inner advantage blend | 0.5 | 0.5 | 0.5 |
| $\beta_{\text{KL}}$ | KL to reference | 0.01 | 0.01 | 0.01 |
| $\epsilon$ | PPO clip | 0.2 | 0.2 | 0.2 |
| $T$ | InfoNCE temp | — | 0.1 | 0.1 |
| $k$ | Comm graph degree | 4 | 4 | 4 |
| $\tau$ | Gate threshold | 0.5 | 0.5 | 0.5 |
| $W$ | Leaf-recency window | 4 | 4 | 4 |
| $\beta$ | Completion bonus | 1.0 | 1.0 | 1.0 |
| $\alpha_k$ | Per-slot reward | 0.2 | 0.2 | 0.2 |
| $\gamma$ | Budget penalty | 0.3 | 0.3 | 0.3 |
| $T_{\max}$ | Rollout step cap | 32 | 32 | 32 |
| LoRA rank | — | 16 | 16 | 16 |
| Learning rate | — | 4e-5 | 4e-5 | 4e-5 |

LoRA target: attention projections + MLP up/down. Base frozen.

---

## 6. Measurements and Diagnostics

### 6.1 Per-step logging

- Task success rate (held-out).
- Per-slot fill rate.
- **Productive-cascade ratio**: $|\text{productive}| / |\text{all causal cascades}|$.
- $P(s_{\text{productive}})$, $P(s_{\text{causal}})$: histograms.
- $P(T_{\text{productive}})$, $P(T_{\text{causal}})$: duration distributions.
- Mean broadcasts per rollout, mean gate value.
- Budget utilization $B_{\text{used}}/B$.
- $\hat{I}(z_{B_p}; z_{B_c})$ for productive-cascade edges.
- Distribution of attribution values $A_i$ across agents (Gini, max/median ratio).

### 6.2 Periodic deep diagnostics

Every $K=500$ steps:

- **Power-law fit** on $P(s_{\text{productive}})$: ML fit per Clauset–Shalizi–Newman, p-value via bootstrap, comparison vs lognormal/exponential.
- **Duration exponent fit** on $P(T_{\text{productive}})$.
- **Self-consistency**: $\langle s\rangle(T)$ slope $\gamma_{\text{measured}}$; verify $|\gamma_{\text{measured}} - (\alpha-1)/(\tau-1)| < 0.3$.
- **MI-vs-graph-distance**: $\hat{I}$ as function of graph distance $d_\mathcal{G}(i,j)$; should decay as power law at criticality.

### 6.3 Phase 3: Finite-size scaling (measurement only)

Post-Phase-3:

1. Inference rollouts at $N \in \{32, 64, 128, 256\}$ with all other hyperparameters fixed.
2. Compute $P(s_{\text{productive}}; N)$ per $N$.
3. Test data collapse: under scaling ansatz $P(s; N) = s^{-\tau} f(s/N^{1/\sigma})$, plot $s^\tau P(s; N)$ vs $s/N^{1/\sigma}$. Critical $\Rightarrow$ all $N$ collapse onto a single $f$.
4. Collapse residual = MSE between rescaled curves. Success: $< 0.15$.

### 6.4 v1 Success criteria

**Success** if all hold:
1. Trained swarm beats no-comm baseline by $\geq 25\%$ on efficiency-weighted task success.
2. $P(s_{\text{productive}})$ fits power law with $\tau \in [1.3, 2.5]$ over $\geq 1.5$ decades, p-value $\geq 0.1$.
3. Self-consistency $|\gamma - (\alpha-1)/(\tau-1)| < 0.3$.
4. Data collapse residual $< 0.15$.

**Partial success** (still publishable):
- (1) holds, (2)-(4) fail → criticality terms didn't induce criticality but did help coordination.
- (2)-(4) hold, (1) fails → induced criticality without computational benefit. Calls hypothesis into question for LLM swarms.

**Failure**: neither task improvement nor critical statistics. Loss formulation needs revision.

---

## 7. Baselines and Ablations

| Run | Description | Tests |
|---|---|---|
| B1 | Single agent, full context (where corpus fits) | Upper bound |
| B2 | $N$ independent agents, no comm, sharded | Pure-parallel baseline |
| B3 | Swarm, Phase 1 only ($\lambda_{\text{MI}} = 0$) | Does MI loss matter? |
| B4 | Swarm, Phase 2, no sparsity ($\lambda_{\text{sp}} = 0$) | Does sparsity matter? |
| B5 | Swarm, MI over *all* causal cascades (not productive-only) | Does productive restriction prevent noise-MI? |
| B6 | Swarm, fully connected graph | Does sparse topology matter, or just gating? |
| B7 | Swarm, no shared parameters (per-agent LoRA) | Confirms shared-policy is necessary for FSS |
| B8 | Swarm, uniform attribution (binary on/off, no DAG-prop) | Does recursive attribution improve over flat? |
| B9 | Swarm, leaf identification via counterfactual (§3.6 alt) | Cheap leaf method robust enough? |

B3, B5, B8 are the critical scientific ablations.

---

## 8. Known Risks and Open Questions

**Risk: Productive cascades sparse early on.** If swarm hasn't learned slot-fills in the first hundreds of rollouts, MI term has no signal, training stalls. *Mitigations*: Phase 0 single-agent warmup; monitor productive-cascade rate; if $<1\%$ for $>500$ steps, fall back to brief curriculum where productive restriction relaxed.

**Risk: Free-riding under shared rewards.** Recursive attribution + participation weights is a cheap proxy. If empirical agent-contribution variance stays low (most agents at $w_i \approx 0.5$), escalate to true counterfactual difference rewards.

**Risk: Counterfactual gate logits expensive.** v1 subsamples to 25% during training. May need further reduction or distillation of the counterfactual signal into a learned predictor.

**Risk: Cheap leaf identification is noisy.** False positives (broadcasts referenced by reporter but not actually load-bearing) inflate productive-cascade count and dilute attribution. *Diagnostic*: periodic sanity check using counterfactual leaves on a subsample, compare leaf overlap. *Escalation*: switch to counterfactual leaves (§3.6 alternative) if false-positive rate is high.

**Risk: Attribution diluted to insignificance for deep cascades.** Width discounting plus depth discounting can drive $A(B)$ to very small values for high-fan-in deep ancestors. *Mitigation*: monitor the distribution of $A(B)$; if dominated by leaves with negligible upstream signal, consider a softer discount (e.g. per-parent weight $1/\sqrt{|\text{parents}|}$). Defer to v1.5.

**Risk: Block-level credit normalization off.** Long CoTs might still dominate gradient norms despite length normalization. *Diagnostic*: per-block-type gradient-norm logging.

**Open: Does shared LoRA + heterogeneous shards produce identical agents?** *Diagnostic*: input-permutation test — at evaluation, swap two agents' shards mid-rollout; verify swarm performance unchanged.

**Open: Long-rollout CoT growth.** $T_{\max}=32$ swarm-steps × multi-block agent turns can produce large contexts. v1: sliding-window truncation. v2: explicit summarization.

**Open: Tinker rate limits and async coordination.** $N=64$ concurrent `sample()` calls per swarm-step plus counterfactual passes is a substantial throughput. May need rate-limit-aware batching or staggered stepping.

---

## 9. What v1 Does Not Cover

- **Vuln fuzzing application.** Reserved for v2 once toy task validates.
- **Schema-blind agents.** v1 uses question-driven framing; discovery-only setup is v2.
- **Dynamic semantic-locality topology.** v1 = static random k-regular; v2 = sentence-transformer-driven dynamic neighborhoods.
- **Learned communication parameters.** v1 fixes $k, \tau$; v2 learns them.
- **Hierarchical baselines** (Kimi-K2.5-style PARL). Important for positioning, not necessary for core claim.
- **Theoretical analysis.** Empirical-first; theory in v3 if empirics work.

---

## 10. Implementation Plan

1. Build corpus + composite-needle generator at 5M-token scale. (Day 1–3)
2. Match function + per-slot reward + paraphrase calibration. (Day 1–2)
3. Phase 0 single-agent baseline via Tinker. (Day 4–5)
4. Swarm rollout loop: shared LoRA, k-regular graph, tool-call action mapping, async per-agent sampling. (Day 6–10)
5. Causal-DAG construction with counterfactual gate logits + cheap leaf identification. Validate on synthetic ground-truth rollouts. (Day 11–13)
6. Recursive attribution propagation. Unit tests against worked examples (§3.7). (Day 13–14)
7. GRPO with nested-group advantages + participation weights, via `forward_backward_custom`. Validate Phase 1. (Day 15–18)
8. InfoNCE bilinear critic + sparsity loss. Validate Phase 2. (Day 19–23)
9. Multi-scale evaluation + FSS measurement. Validate Phase 3. (Day 24–28)
10. Ablations B1–B9. (Day 29–34)
11. Write up. (Day 35–40)

Total: ~6 weeks single-person, accounting for RL-debugging friction. Compute scales with $G \times N \times L_C \times T_{\max}$; Tinker handles distribution.

---

## Appendix A — Tinker integration notes

**Group structure.** `EnvGroupBuilder.make_envs()` returns the $N$ agent-envs for one rollout. The training loop iterates over $G$ such builders per step. Tinker's built-in advantage centering operates over the outer group; we override with our two-level (inner+outer) computation in a custom advantage function.

**Custom loss.** `forward_backward_custom` takes a function `(model_outputs, datum) → scalar`. We pass per-block advantages and per-block log-probs in `Datum.metadata`; the loss function reconstructs the clipped surrogate plus KL, with block-length normalization applied as in §3.4.

**InfoNCE side-channel.** The bilinear critic is a separate PyTorch module on the client side. During rollouts, we record residual activations at broadcast tokens via `sample()`'s logits-and-hidden-state hook (Tinker exposes hidden states for the final layer). The critic trains locally; its gradient is composed with the policy gradient via a second `forward_backward_custom` call where the residuals act as policy outputs and the loss is the InfoNCE objective.

**Counterfactual gate logits.** For each broadcast, we re-run `sample()` with a modified message buffer (the candidate parent broadcast removed). Cheap because we only need the gate logit (one token's logit) — but we still pay one forward pass per (broadcast, candidate-parent) pair. Subsample to 25% in training as noted in §3.5.

**Checkpointing.** `save_state()` / `load_state()` for full training-state snapshots; `save_weights_for_sampler()` for inference-only weights at evaluation. Save every 100 steps; keep 5 most recent.

**Models to try, in order**: Qwen-2.5 3B (fast iteration), Qwen-2.5 7B (v1 target), Llama-3.2 8B (cross-check). LoRA rank 16 across the board per Tinker's "LoRA Without Regret" guidance — they argue rank 16–32 matches full FT for RL workloads.
