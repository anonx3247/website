# $\epsilon$-Machines: The Basis of Mechanistic Interpretability

I would like to express and formalize, in part, a theory (or intersection of theories) I've developed regarding a possible path forward in mechanistic interpretability.

Mechanistic Interpretability aims to explain the *mechanics* of AI models—not simply *that* a model produces a certain output, but *why* it produces that output through understanding the underlying computational process. It's an intuitive goal: the entire field of mathematics and formal proof theory emerged from the need to understand the reasoning used by others to reach conclusions. Any computation can be viewed as algorithmic, and neural networks should likewise follow algorithmic, causally-modeled principles.

Causality is thus the core feature of mechanistic interpretability: given inputs, why did the model output what it did? Many methods have been developed to investigate this question—from sparse autoencoders to Anthropic's Natural Language Autoencoders^1. These approaches work with activations in model layers and attempt to reconstruct the computations that gave rise to them. They've successfully identified the geometric structures underlying newline prediction in language models^2, uncovered the circuits responsible for arithmetic in LLMs^3, and isolated basic computational operations in simple logical systems^4.

For this work, we examine computation through two lenses: algebraic and statistical.

Gauderis et al.^5 propose that a sound mechanistic interpretability framework minimizes complexity while maintaining fidelity to the underlying computation. This mirrors how physics works: projectile motion abstracts away object volume and shape in favor of vector calculus derived from F=ma. Similarly, thermodynamics uses group statistics to circumvent the intractability of modeling individual particles—we focus on empirical laws (adiabatic equations for gases, the two laws of thermodynamics) rather than attempting to track trillions of atoms. Mechanistic interpretability follows this principle: we aim to reason about the model's high-level "thinking process" with sufficient predictive power to understand why it will succeed or fail on various tasks before deployment. This predictability is particularly valuable for alignment research, allowing us to characterize model capabilities in broad terms without knowing every possible user input.

## An Algebraic View of Neural Networks

One way to conceptualize neural networks is as quasi-algebraic systems. The fundamental operation of modern neural networks—matrix multiplication—is inherently algebraic. Though activation functions add complexity, neural networks remain fundamentally algebraic, forming what we can understand as a monoid under function composition.

A transformer block exemplifies this structure:

$$\text{output} = \text{attention} \circ \text{linear} \circ \text{normalize} \circ \text{input}$$

This is naturally interpretable as monoid composition.

Many are familiar with groups—consider addition over real numbers, $(\mathbb{R}, +)$, where the operation maps two group elements to an output that remains in the group. Groups have one property monoids lack: every element has an inverse. For any $a \in \mathbb{R}$, there exists $-a$ such that $a + (-a) = 0$. Monoids lack this requirement, which makes sense for neural networks: function composition is not generally invertible. We cannot reverse many neural operations—softmax operations, for instance, sum row elements, destroying information that cannot be recovered from the output alone.

We don't need to model the actual monoid representing the neural network directly—it would be as complex and noisy as the network itself. Instead, we seek an underlying, simpler monoid (in a sense we'll make precise) that still produces approximately the same outputs with manageable error.

Category theory provides the most general framework for understanding natural transformations. A related field, representation theory, studies how algebraic structures can be canonically embedded (up to isomorphism) into vector spaces. This offers a principled way to understand how abstract algebraic operations appear in neural network weight matrices—the fundamental substrates of neural computation.

A *representation* of a monoid $M$ is a homomorphism:

$$\rho: M \to \text{GL}_d(\mathbb{R})$$

In practical terms: each element $m \in M$ is assigned a matrix $\rho(m) \in \mathbb{R}^{d \times d}$ such that:

$$\rho(m_1 \cdot m_2) = \rho(m_1) \cdot \rho(m_2)$$

Irreducible representations offer particular insight. When a monoid representation is decomposed into irreducibles, it takes a block-diagonal form:

$$\rho(m) = \begin{pmatrix} \rho_1(m) & 0 & 0 \\ 0 & \rho_2(m) & 0 \\ 0 & 0 & \rho_3(m) \end{pmatrix}$$

This decomposition is dictated by algebraic structure, not by choice. It reveals causally independent pathways in network activations—channels that do not interact with one another. This addresses a limitation in applying singular learning theory to neural networks: when we measure an activation, we typically cannot localize it to a single neuron or layer, as patterns often distribute across many layers, each processing multiple intertwined patterns. Decomposition into irreducible representations offers a mechanism for achieving genuine locality in this sense.

Since we care about capturing an abstract-level monoid distinct from the noisy actual network, we work with approximate representations, permitting an error term. The magnitude of this error quantifies how abstract our monoid is relative to the network.

### Relation to Platonic Representation Theory

A recurring question in interpretability circles concerns whether intelligent systems exhibit convergent evolution toward canonical concepts and abstractions—what is sometimes called "Platonic" structure, in reference to Plato's *World of Ideas*, where concepts exist independently of their physical manifestations.

If such convergence exists, it would have profound implications. It would enable unified understanding of both biological and artificial intelligence. It would also address a significant alignment challenge: if I specify a model's values using, say, Asimov's Three Laws of Robotics in constitutional AI, how can I verify that the model's internal representation of these terms matches mine? This difficulty exists in human communication too, but we typically succeed because we are "close enough." If the convergence hypothesis holds, we might hope that models similarly converge to sufficiently aligned internal representations.

This view entails that mechanistic models of LLMs work across architectures—that the substrate (biological, RNN, or transformer) matters less than the underlying computation. We see evidence in convergent concepts across LLMs from different laboratories and in architecture-agnostic theories of learning like the information bottleneck^6.

Monoid-based representations are powerful precisely because they abstract computations independent of substrate or architecture. Moreover, by identifying canonical operations (rather than arbitrary ones) through $\epsilon$-machines, we can establish that multiple architectures execute the same high-level computation despite differing low-level circuit implementations.

## Statistical Mechanics and $\epsilon$-Machines

Crutchfield & Shalizi^7 established computational mechanics as a field intermediating between statistical mechanics and computation. Particularly for systems near criticality, computational mechanics identifies processes on the Pareto frontier—exhibiting structure (hence non-random) while remaining complex at the base level (and thus not fully reducible to components). This makes computational mechanics especially suited to mechanistic interpretability.

Formally, they study stationary stochastic processes with infinite pasts. Let $(X_t)_{t \in \mathbb{Z}}$ be such a process with alphabet $\mathcal{A}$:

- **Past:** $X^{\leftarrow} = \ldots X_{-2} X_{-1} X_0$ (observations up to and including time 0)
- **Future:** $X^{\rightarrow} = X_1 X_2 X_3 \ldots$ (observations from time 1 onward)
- **Finite sequences:** $x^n = x_0 x_1 \ldots x_{n-1}$ denotes a sequence of length $n$

**Definition (Causal Equivalence):** Two semi-infinite pasts $x^{\leftarrow}$ and $x'^{\leftarrow}$ are causally equivalent if they produce identical future distributions:

$$P(X^{\rightarrow} | X^{\leftarrow} = x^{\leftarrow}) = P(X^{\rightarrow} | X^{\leftarrow} = x'^{\leftarrow})$$

The future "looks the same" regardless of which past you had. We partition all pasts into equivalence classes under this relation, calling these classes **causal states** $s \in \mathcal{S}$. Transitions between states follow:

$$T(s_{t+1} | s_t, a) = P(S_{t+1} = s_{t+1} | S_t = s_t, X_t = a)$$

This formalism parallels reinforcement learning and Markov decision processes, making it particularly useful for understanding autoregressive models, which are themselves naturally cast as MDPs.

The **$\epsilon$-machine** is formally defined as:

$$\varepsilon = (\mathcal{S}, \mathcal{A}, T, P, s_0)$$

where:
- $\mathcal{S}$: the causal state space
- $\mathcal{A}$: the output alphabet
- $T(s' | s, a)$: transition probabilities between states
- $P(a | s)$: emission probabilities (output conditioned on current state)
- $s_0$: the initial state distribution

This is a hidden Markov model with a defining property: it is the unique minimal HMM generating the process.

### Theorem (Crutchfield & Shalizi, 2003)^7

Among all HMMs that generate a process $(X_t)$, the $\epsilon$-machine possesses the smallest state space $|\mathcal{S}|$.

Therefore, if two processes have isomorphic $\epsilon$-machines, they share the *same fundamental computational structure*, even if they appear superficially different (different alphabets, rates, or implementations). The $\epsilon$-machine also minimizes entropy and free energy—maximizing economy of description—precisely solving the faithfulness-complexity tradeoff identified by Gauderis et al.^5

We apply this to neural activations, treating layer outputs as trajectories:

$$\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_T$$

The $\epsilon$-machine of this sequence represents the minimal computational structure the layer implements.

## Bringing the Two Together

We have discussed monoid representation and computational mechanics separately. Their integration follows naturally: the transition set of an $\epsilon$-machine forms a monoid under composition:

$$M = \langle T(\cdot | s, a) : s \in \mathcal{S}, a \in \mathcal{A} \rangle$$

Earlier we motivated the search for algebraic structure, but choosing which structure was unclear. Now we see that the $\epsilon$-machine is the ideal candidate: it minimizes the tradeoff between faithfulness and complexity, grounding our choice in information theory and dynamics rather than heuristic preference.

## Further Work and Questions Left

Many aspects of this theory remain to be developed. How do we reconstruct the underlying monoid from a neural network? Does a practical algorithm exist? How do we choose appropriate levels of abstraction and set the error term tolerances? Will we retain canonicity if error terms are necessary? (This question is critical: allowing error relaxes uniqueness, leaving multiple candidates. How should we choose among them?)

Assuming we can extract a network's $\epsilon$-machine, what next? One avenue involves using its representation to apply canonical transformations to weight matrices, producing maximally interpretable forms with localized activations—facilitating analysis across various inputs. Another involves constructing causal DAGs from the monoid structure and using these to predict typical outputs for new inputs. We might classify input types by their causal behavior within the monoid, creating equivalence classes of semantically similar inputs. Finally, detecting misalignment, jailbreaks, and dangerous requests might become tractable through monitoring decorrelation and inconsistency between internal representation and model outputs.

---

## References

^1. Templeton, A., et al. (2024). "Scaling Monosemanticity: Interpreting Neural Networks with Dictionary Learning." ArXiv.

^2. Nanda, N., McGrath, T., et al. (2025). "When Models Manipulate Manifolds: The Geometry of a Counting Task." Transformer Circuits Thread. https://transformer-circuits.pub/2025/linebreaks/index.html

^3. Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). "Progress Measures for Grokking via Mechanistic Interpretability." ArXiv:2301.05217.

^4. Crutchfield, J. P., & Hanson, J. E. (1997). "Computational mechanics of cellular automata: An example." *Physica D*, 103(2–4), 169–189. [Crutchfield's work on extracting $\epsilon$-machines from simple automata provides foundational methodology for toy symbolic systems.]

^5. Gauderis, W., Dooms, T., Homer, S. T., Ayonrinde, K., & Wiggins, G. A. (2025). "From Mechanistic to Compositional Interpretability." ArXiv:2605.08934.

^6. Tishby, N., & Schwartz-Ziv, R. (2015). "Opening the Black Box of Deep Neural Networks via Information." *Journal of Statistical Physics*, 162(7), 1391–1410. ArXiv:1503.06365.

^7. Crutchfield, J. P., & Shalizi, C. R. (2003). "Regularities Unseen, Randomness Observed: Levels of Description in Complex Systems." *Journal of Chemical Physics*, 104(25), 1–77.
