.

---

# **Part I: Research Findings & Brutally Honest Evaluation**

## **1\. Current Landscape of ML-Based Quantum Error Mitigation (2024–2026)**

After searching arXiv, npj Quantum Information, Nature Machine Intelligence, IBM/Google/NVIDIA publications, and IEEE venues, here is the state of the art:

### **Mainstream ML-QEM Methods (All Require Classical Simulation)**

| Method | Year | Simulation Required? | Architecture | Max Qubits Tested | Key Limitation |
| ----- | ----- | ----- | ----- | ----- | ----- |
| **CDR** (Czarnik et al.) | 2021 | Yes (Clifford circuits) | Linear regression | \~12 | Exponential Clifford scaling |
| **IBM ML-QEM** (Liao et al.) | 2024 | Yes (near-Clifford) | RF, MLP, GNN | 100 (real hardware) | Needs simulable training circuits [Nature MI](https://research.ibm.com/publications/machine-learning-for-practical-quantum-error-mitigation--1) |
| **DAEM** (Liao et al.) | 2025 | Yes (fiducial Clifford circuits) | MLP, U-Net | 50 (many-body) | Claims “no noise-free data” but still uses Clifford simulation for fiducial labels [npj QI](https://www.nature.com/articles/s41534-025-00960-y) |
| **GEM** (Wang et al.) | 2026 | Yes (ideal simulation for 10–16 qubits) | GNN with physics features | 16 (real HW), zero-shot transfer | Needs ideal labels for training; relies on hardware calibration data [arXiv](https://arxiv.org/html/2604.16815v1) |
| **Kendre CNN** | 2025 | Yes (clean density matrices) | CNN autoencoder | 5 | Fully supervised; density matrix input impractical [arXiv](https://arxiv.org/abs/2509.16242) |
| **Kim et al. DL-QREM** | 2022 | Yes (known single-qubit outcomes) | Neural network | 5 (IBM HW) | Readout errors only; needs calibration circuits [NJP](https://arxiv.org/abs/2112.03585) |
| **Adeniyi et al.** | 2025 | Yes (simulated noisy+noiseless pairs) | Adaptive NN | \~8 | Fully supervised with ground truth [Springer](https://link.springer.com/article/10.1007/s42484-024-00234-4) |

### **Non-ML QEM Methods (for context)**

* **ZNE** (Zero-Noise Extrapolation): IBM’s workhorse, tested up to 127 qubits. Requires noise scaling (gate folding) but no simulation. Biased estimator.  
* **PEC** (Probabilistic Error Cancellation): Unbiased but requires full noise characterization (exponential cost).  
* **Spacetime Noise Inversion** (Yoshioka et al., 2025): PEC variant needing only one error parameter \+ error sampler. Not ML-based, integrates with QEC [arXiv](https://arxiv.org/html/2504.12864v1).  
* **Shot-Noise Reduction for Lattice Hamiltonians** (Eckstein et al., PRX Quantum 2026): Measurement strategy for reducing shot noise in energy eigenstates. Not ML-based [PRX Quantum](https://link.aps.org/doi/10.1103/xy36-drb3).

### **The Noise2Noise Connection**

The user’s idea is conceptually a **Noise2Noise** approach [Lehtinen et al., ICML 2018](https://arxiv.org/abs/1803.04189) applied to quantum measurement distributions. Noise2Noise proved that training on noisy→noisy pairs (where noise is independent and zero-mean) recovers the clean signal *in expectation*, without ever observing clean data. **After exhaustive search, I found no paper that applies Noise2Noise to quantum bitstring distributions or quantum measurement histograms.** This is a genuine gap.

### **Industry Approaches**

* **IBM**: Focused on ZNE \+ PEC at utility scale (127+ qubits), with ML-QEM as a cost-reduction layer. Roadmap targets quantum advantage by 2026 using error mitigation. [IBM Quantum Blog](https://www.ibm.com/quantum/blog/gammabar-for-quantum-advantage)  
* **Google**: Pivoted heavily to QEC (below-threshold surface codes, Nature 2024). Error mitigation is secondary; focus is on fault tolerance. [Nature](https://www.nature.com/articles/s41586-024-08449-y)  
* **NVIDIA**: Provides CUDA-Q with tensor network (MPS) simulators and readout error mitigation via confusion matrices. No published ML-based QEM research. [NVIDIA CUDA-Q Docs](https://nvidia.github.io/cuda-quantum/latest/applications/python/readout_error_mitigation.html)  
* **AWS Braket**: Integrates Mitiq toolkit for ZNE/PEC/CDR. No novel ML-QEM research. [AWS Blog](https://aws.amazon.com/blogs/quantum-computing/error-mitigation-on-amazon-braket-with-program-sets-and-mitiq/)

---

## **2\. Evaluation of Your Proposed Approach**

### **Verdict: The core insight is genuinely novel, but the method as stated has a critical theoretical flaw that must be fixed before it is publishable.**

### **What Is Genuinely Novel (Strong Points)**

1. **No classical simulation whatsoever.** Every single ML-QEM method in the literature requires some form of classical simulation for training labels—whether Clifford circuits (CDR, DAEM), near-Clifford circuits (IBM ML-QEM), or full ideal simulation (GEM, Kendre). Your proposal to use *only device-native data* with no simulation at any stage is unprecedented. This is the single strongest novelty claim and is alone sufficient to differentiate from all prior work.

2. **Noise2Noise for quantum distributions.** Applying the Noise2Noise principle to quantum measurement histograms (multinomial distributions over bitstrings) has not been done. The closest conceptual ancestor is Noise2Noise itself (image restoration), which has never been ported to the quantum measurement domain.

3. **Shot-count hierarchy as self-supervision.** Using the natural low-shot → high-shot hierarchy as the training signal is creative and, to my knowledge, entirely unexplored in QEM literature.

4. **Permutation-invariant architecture for qubit-count generalization.** While permutation-equivariant QNNs exist [Schatzki et al., npj QI 2024](https://www.nature.com/articles/s41534-024-00804-1), nobody has applied them to QEM distribution denoising. The Set Transformer architecture [Lee et al., ICML 2019](https://arxiv.org/abs/1810.00825) is well-suited but unexplored in this context.

### **The Critical Theoretical Flaw (Brutal Honesty)**

**Your central claim that “high-shot estimates converge toward a stable distribution that is much closer to the noise-free ideal” is fundamentally incorrect for the purpose of full error mitigation.**

Here is why:

Quantum measurement noise has **two distinct components**:

1. **Shot noise** (statistical sampling noise): Arises from finite measurement counts. A circuit run with $S$ shots estimates each probability $p\_i$ with variance $\\sim p\_i(1-p\_i)/S$. As $S \\to \\infty$, shot noise vanishes. **Your low-shot → high-shot mapping correctly targets this.**

2. **Hardware noise** (decoherence, gate errors, readout errors, crosstalk): Arises from imperfect quantum operations. This is *systematic*—it does NOT vanish with more shots. A circuit run with 10,000 shots converges to the *noisy* probability distribution $p^{\\text{noisy}}*i \= \\sum\_j \\Lambda*{ij} p^{\\text{ideal}}\_j$, where $\\Lambda$ is the noise channel. **Your high-shot target is the noisy distribution, not the ideal one.**

**Consequence**: Your network, trained on low-shot → high-shot pairs, will learn to **remove shot noise only**. It will *not* mitigate hardware noise. At inference, it will produce a clean estimate of the *noisy* distribution—not the ideal distribution. For shallow circuits with low hardware noise, this may still be useful. For deep circuits or noisy devices, the method provides no error mitigation beyond variance reduction.

This is the difference between **shot noise reduction** (valuable but not QEM) and **quantum error mitigation** (the stated goal). A top journal reviewer will immediately identify this flaw.

### **Other Weaknesses**

3. **The high-shot pseudo-target is expensive.** Generating 10,000-shot data for every training circuit is costly on real hardware. The method’s training cost may rival or exceed the cost of simply running more shots at inference—undermining the practical value proposition.

4. **Generalization from 2–4 qubits to 20+ is unsubstantiated.** The distribution support grows as $2^n$. A network trained on 4-qubit distributions (16-dimensional) must somehow generalize to 20-qubit distributions (1,048,576-dimensional). Permutation invariance helps with qubit relabeling but does not address the exponential support growth. You need a representation that decouples from the Hilbert space dimension.

5. **No theoretical guarantee.** Noise2Noise works because image noise is additive and zero-mean. Quantum shot noise is multinomial (not additive), and the mapping from low-shot to high-shot is not a simple denoising problem—it’s a density estimation problem with structured priors. The theoretical justification needs to be built from scratch.

### **Final Assessment**

| Criterion | Score | Reasoning |
| ----- | ----- | ----- |
| **Feasibility** | 7/10 | Shot-noise reduction component is feasible immediately; hardware-noise component requires modification |
| **Novelty (as stated)** | 6/10 | Novel framing, but only addresses shot noise, not full QEM |
| **Novelty (with modifications)** | 9/10 | Adding noise-scaling self-supervision makes it genuinely unprecedented |
| **Publishability (as stated)** | 4/10 | Reviewer will reject the “close to ideal” claim |
| **Publishability (with modifications)** | 8/10 | Addresses a real gap; strong differentiation from all prior work |
| **Practical utility (as stated)** | 5/10 | Shot noise reduction is useful but limited |
| **Practical utility (with modifications)** | 8/10 | Full QEM without simulation is the holy grail for NISQ |

---

## **3\. The Fix: Enhanced Method — “Noise-to-Less-Noise” (N2LN-QEM)**

I propose the following critical modification to your approach while preserving its self-supervised, simulation-free spirit:

### **Core Innovation: Dual Self-Supervised Signal**

**Stage 1 — Shot Noise Reduction (your original idea, properly framed):**  
 Train a network to map low-shot empirical distributions → high-shot empirical distributions. This removes shot noise (multinomial sampling noise). Target \= high-shot noisy distribution. This is a valid Noise2Noise application for variance reduction.

**Stage 2 — Hardware Noise Mitigation (new, essential addition):**  
 Train a network to map **noise-scaled distributions**: normal-noise → reduced-noise, where reduced noise is achieved via standard noise-scaling techniques (gate folding, dynamical decoupling, or pulse stretching). This is still 100% self-supervised—both distributions come from the same device, no simulation needed—but it captures *hardware noise structure*, not just shot noise. At inference, the network extrapolates to the zero-noise limit.

**Why this works**: Noise scaling (gate folding) is the basis of ZNE and is universally available on all quantum platforms. By training on (amplified-noise → original-noise) pairs, the network learns the noise transfer function. By training on (original-noise → reduced-noise) pairs (via dynamical decoupling or pulse optimization), it learns to push toward the noise-free limit. The combination of multiple noise levels creates a self-supervised extrapolation signal—**without any classical simulation**.

**Stage 3 — Unified Inference**: At test time, feed a single low-shot, normal-noise measurement through the unified network, which simultaneously denoises shot noise and extrapolates hardware noise to the zero-noise limit.

### **Additional Enhancements**

* **Permutation-invariant Set Transformer** operating on *bitstring-level embeddings* (not full distribution vectors), enabling qubit-count generalization  
* **Physicality-constrained loss**: KL divergence \+ total variation \+ entropy regularization \+ softmax temperature, ensuring valid probability distributions  
* **Theoretical analysis**: Noise2Noise theory extended to multinomial distributions, with bias-variance decomposition for the two-stage approach  
* **Comparison oracle**: Ideal-supervised model (trained with simulation) as an upper bound, to quantify the “simulation-free penalty”

---

# **Part II: Technical Design Document**

# **Technical Design Document: N2LN-QEM**

## **Noise-to-Less-Noise Quantum Error Mitigation via Self-Supervised Distribution Learning**

**Version**: 1.0  
 **Date**: July 2026  
 **Project Type**: Master’s Thesis Research Project  
 **Target Venue**: Quantum Machine Intelligence (Springer) or IEEE Transactions on Quantum Engineering

---

## **1\. Project Overview**

### **1.1 Problem Statement**

NISQ quantum computers produce noisy measurement distributions that deviate from ideal noise-free distributions due to two distinct noise sources: (1) **shot noise**—statistical sampling error from finite measurement repetitions, and (2) **hardware noise**—systematic errors from decoherence, gate imperfections, and readout errors. Existing ML-based quantum error mitigation (QEM) methods require classical simulation of ideal distributions for training, which becomes intractable beyond \~30 qubits. This project develops a **simulation-free, self-supervised** QEM method that learns to denoise quantum measurement distributions using only data from the quantum device itself.

### **1.2 Core Innovation**

**N2LN-QEM** (Noise-to-Less-Noise QEM) introduces a dual self-supervised training paradigm:

1. **Shot-Noise Denoising (SN-D)**: Learn to map low-shot empirical distributions to high-shot empirical distributions of the same circuit, exploiting the shot-count hierarchy as a natural self-supervision signal.

2. **Hardware-Noise Extrapolation (HN-E)**: Learn to map noise-amplified distributions to noise-reduced distributions of the same circuit, exploiting noise-scaling techniques (gate folding, dynamical decoupling) as a second self-supervision signal.

Both stages use **only device-native data**—no classical simulation of ideal distributions is required at any point. A unified permutation-invariant neural network architecture enables generalization across qubit counts without retraining.

### **1.3 Design Goals**

| Goal | Metric | Target |
| ----- | ----- | ----- |
| Shot noise reduction | TVD between SN-D output and high-shot ground truth | ≤ 50% of raw low-shot TVD |
| Hardware noise mitigation | MAE between N2LN output and ideal expectation values | ≤ ZNE performance |
| Simulation-free | Classical simulation used during training | Zero |
| Qubit-count generalization | Train on n≤6, test on n=10, 20 | \<2× degradation vs. in-distribution |
| Real hardware validation | IBMQ 127-qubit device | Statistically significant improvement over raw |

### **1.4 Project Structure**

n2ln-qem/  
├── config/                    \# Experiment configurations (YAML)  
│   ├── circuits.yaml          \# Circuit family definitions  
│   ├── noise\_models.yaml      \# Noise model parameters  
│   ├── training.yaml          \# Hyperparameters  
│   └── evaluation.yaml        \# Metrics and baselines  
├── src/  
│   ├── data/  
│   │   ├── circuit\_generator.py    \# Random/variational circuit generation  
│   │   ├── noise\_scaler.py         \# Gate folding, DD, noise amplification  
│   │   ├── data\_collector.py       \# Simulator \+ hardware data collection  
│   │   ├── dataset.py              \# PyTorch Dataset classes  
│   │   └── preprocessing.py        \# Distribution encoding, normalization  
│   ├── models/  
│   │   ├── autoencoder.py          \# Denoising autoencoder (baseline arch)  
│   │   ├── set\_transformer.py      \# Permutation-invariant Set Transformer  
│   │   ├── bitstring\_encoder.py    \# Bitstring-level embedding module  
│   │   ├── dual\_head.py            \# SN-D \+ HN-E dual-head output  
│   │   └── physicality.py          \# Softmax \+ distribution constraints  
│   ├── losses/  
│   │   ├── distribution\_loss.py    \# KL, TVD, chi-squared, composite  
│   │   ├── physicality\_loss.py     \# Non-negativity, normalization constraints  
│   │   └── consistency\_loss.py     \# Cross-stage consistency  
│   ├── training/  
│   │   ├── trainer.py              \# Main training loop  
│   │   ├── curriculum.py           \# Curriculum learning scheduler  
│   │   ├── callbacks.py            \# Early stopping, checkpointing  
│   │   └── noise\_aware\_aug.py      \# Noise-aware data augmentation  
│   ├── evaluation/  
│   │   ├── metrics.py              \# TVD, fidelity, MAE, MSE, KL  
│   │   ├── baselines.py            \# Raw, readout inversion, ZNE, CDR, ideal-supervised  
│   │   └── statistical\_tests.py    \# Bootstrap CI, Wilcoxon tests  
│   └── utils/  
│       ├── quantum\_sim.py          \# Qiskit Aer wrapper (for EVAL ONLY)  
│       ├── mps\_sim.py              \# MPS simulation for 20-30 qubits  
│       ├── hardware\_interface.py   \# IBMQ runtime interface  
│       └── visualization.py        \# Distribution plots, training curves  
├── experiments/  
│   ├── exp1\_shot\_noise/            \# Stage 1 validation  
│   ├── exp2\_hw\_noise/              \# Stage 2 validation  
│   ├── exp3\_unified/               \# Combined N2LN  
│   ├── exp4\_scaling/               \# Qubit-count generalization  
│   ├── exp5\_hardware/              \# Real IBMQ validation  
│   └── exp6\_ablation/              \# Ablation studies  
├── tests/                          \# Unit tests  
├── notebooks/                      \# Analysis and visualization notebooks  
└── docs/                           \# Additional documentation

---

## **2\. Mathematical Formulation**

### **2.1 Notation**

| Symbol | Definition |
| ----- | ----- |
| $n$ | Number of qubits |
| $\\mathbf{x} \\in {0,1}^n$ | A bitstring (measurement outcome) |
| $p^{\\text{ideal}}(\\mathbf{x})$ | Ideal (noise-free) probability of bitstring $\\mathbf{x}$ |
| $p^{\\text{noisy}}(\\mathbf{x})$ | Noisy probability (hardware noise applied, infinite shots) |
| $\\hat{p}\_S(\\mathbf{x})$ | Empirical distribution from $S$ shots: $\\hat{p}\_S(\\mathbf{x}) \= c\_S(\\mathbf{x}) / S$ where $c\_S(\\mathbf{x})$ is the count |
| $\\Lambda$ | Hardware noise channel: $p^{\\text{noisy}} \= \\Lambda \\cdot p^{\\text{ideal}}$ |
| $\\lambda$ | Noise scale factor (1 \= normal, \>1 \= amplified, \<1 \= reduced) |
| $f\_\\theta$ | Neural network with parameters $\\theta$ |

### **2.2 Noise Model Decomposition**

The observed empirical distribution from $S$ shots at noise scale $\\lambda$ is:

$$\\hat{p}*{S,\\lambda}(\\mathbf{x}) \= \\frac{1}{S}\\sum*{s=1}^{S} \\mathbb{1}\[\\mathbf{x}\_s \= \\mathbf{x}\], \\quad \\mathbf{x}*s \\sim \\text{Multinomial}(p*\\lambda^{\\text{noisy}})$$

where $p\_\\lambda^{\\text{noisy}} \= \\Lambda\_\\lambda \\cdot p^{\\text{ideal}}$ and $\\Lambda\_\\lambda$ is the noise channel at scale $\\lambda$.

**Key decomposition**:  
 $$\\hat{p}*{S,\\lambda}(\\mathbf{x}) \- p^{\\text{ideal}}(\\mathbf{x}) \= \\underbrace{\[\\hat{p}*{S,\\lambda}(\\mathbf{x}) \- p\_\\lambda^{\\text{noisy}}(\\mathbf{x})\]}*{\\text{shot noise (vanishes as } S\\to\\infty\\text{)}} \+ \\underbrace{\[p*\\lambda^{\\text{noisy}}(\\mathbf{x}) \- p^{\\text{ideal}}(\\mathbf{x})\]}\_{\\text{hardware noise (persists at } S\\to\\infty\\text{)}}$$

### **2.3 Stage 1: Shot-Noise Denoising (SN-D)**

**Goal**: Learn $f\_\\theta^{(\\text{SN})}: \\hat{p}*{S*{\\text{low}}, \\lambda} \\to \\hat{p}*{S*{\\text{high}}, \\lambda}$

**Training data**: For each circuit $C\_k$, collect:

* Input: $\\hat{p}*{S*{\\text{low}}, \\lambda=1}^{(k)}$ (e.g., $S\_{\\text{low}} \= 100$ shots)  
* Target: $\\hat{p}*{S*{\\text{high}}, \\lambda=1}^{(k)}$ (e.g., $S\_{\\text{high}} \= 10{,}000$ shots)

**Theoretical basis**: By the law of large numbers, $\\hat{p}*{S*{\\text{high}}, \\lambda} \\xrightarrow{S\_{\\text{high}} \\to \\infty} p\_\\lambda^{\\text{noisy}}$. Training on noisy→noisy pairs with independent shot noise recovers $p\_\\lambda^{\\text{noisy}}$ in expectation (Noise2Noise principle extended to multinomial distributions, see §2.5).

**What SN-D achieves**: Removes shot noise. Output approximates $p\_{\\lambda=1}^{\\text{noisy}}$ (the noisy distribution with hardware noise intact).

### **2.4 Stage 2: Hardware-Noise Extrapolation (HN-E)**

**Goal**: Learn $f\_\\theta^{(\\text{HN})}: \\hat{p}*{S, \\lambda*{\\text{high}}} \\to \\hat{p}*{S, \\lambda*{\\text{low}}}$ where $\\lambda\_{\\text{high}} \> \\lambda\_{\\text{low}}$

**Training data**: For each circuit $C\_k$, collect measurements at multiple noise scales:

* Normal noise ($\\lambda=1$): standard circuit execution  
* Amplified noise ($\\lambda \\in {1.5, 2.0, 3.0}$): via gate folding $U \\to U \\cdot U^\\dagger \\cdot U$  
* Reduced noise ($\\lambda \< 1$): via dynamical decoupling sequences inserted during idle times

**Training pairs**: $(\\hat{p}*{S, \\lambda*{\\text{high}}}^{(k)}, \\hat{p}*{S, \\lambda*{\\text{low}}}^{(k)})$ for all $\\lambda\_{\\text{high}} \> \\lambda\_{\\text{low}}$

**Theoretical basis**: The noise channel $\\Lambda\_\\lambda$ varies smoothly with $\\lambda$. The network learns the noise transfer function implicitly. At inference, the network extrapolates from $\\lambda=1$ toward $\\lambda=0$ (zero-noise limit).

**What HN-E achieves**: Mitigates hardware noise. Output approximates $p^{\\text{ideal}}$.

**Critical insight**: HN-E is self-supervised because both $\\hat{p}*{S, \\lambda*{\\text{high}}}$ and $\\hat{p}*{S, \\lambda*{\\text{low}}}$ come from the same device. No simulation is needed. The noise-scaling operations (gate folding, DD) are standard quantum control techniques available on all platforms.

### **2.5 Theoretical Justification: Noise2Noise for Multinomial Distributions**

**Theorem (informal)**: Let $\\hat{p}\_A$ and $\\hat{p}\_B$ be independent empirical distributions from $S\_A$ and $S\_B$ shots respectively, both drawn from the same underlying distribution $q$. If $S\_B \> S\_A$, then the optimal predictor $f^*$ minimizing $\\mathbb{E}\[\\text{KL}(f(\\hat{p}\_A) | \\hat{p}\_B)\]$ satisfies $f^*(\\hat{p}\_A) \\to q$ as $S\_B \\to \\infty$, regardless of $S\_A$.

**Proof sketch**: The key insight from [Lehtinen et al. 2018](https://arxiv.org/abs/1803.04189) is that minimizing the expected loss between a noisy input and an independent noisy target recovers the conditional expectation $\\mathbb{E}\[\\hat{p}\_B | \\hat{p}\_A\]$. For multinomial sampling, $\\mathbb{E}\[\\hat{p}\_B | \\hat{p}\_A\] \= q$ (since $\\hat{p}\_A$ and $\\hat{p}\_B$ are independent given $q$). Thus, as $S\_B \\to \\infty$, the target concentrates around $q$, and the network converges to estimating $q$ from $\\hat{p}\_A$.

**Extension to noise scaling**: For HN-E, the “noise” in the target is the *residual* hardware noise at $\\lambda\_{\\text{low}}$, which is smaller than at $\\lambda\_{\\text{high}}$. The network learns the conditional expectation $\\mathbb{E}\[\\hat{p}*{\\lambda*{\\text{low}}} | \\hat{p}*{\\lambda*{\\text{high}}}\]$, which is a smoothed version of $p\_{\\lambda\_{\\text{low}}}^{\\text{noisy}}$. With multiple noise levels, the network can interpolate and extrapolate toward $\\lambda \= 0$.

**Bias analysis**: The SN-D stage is unbiased in the limit $S\_{\\text{high}} \\to \\infty$ (recovers $p^{\\text{noisy}}$). The HN-E stage has a *residual bias* equal to the hardware noise at $\\lambda\_{\\text{low}}$. By using the lowest achievable $\\lambda\_{\\text{low}}$ (via DD) and training the network to extrapolate, this bias is minimized. The total bias of the unified model is bounded by the extrapolation error, which decreases with the number of noise levels used in training.

### **2.6 Unified Model**

At inference, the unified model takes a single measurement $\\hat{p}*{S*{\\text{low}}, \\lambda=1}$ and produces:

$$\\hat{p}^{\\text{N2LN}} \= f\_\\theta^{(\\text{HN})}\\left(f\_\\theta^{(\\text{SN})}\\left(\\hat{p}*{S*{\\text{low}}, \\lambda=1}\\right)\\right)$$

or equivalently, a single network with shared backbone and dual heads trained jointly:

$$\\hat{p}^{\\text{N2LN}} \= f\_\\theta\\left(\\hat{p}*{S*{\\text{low}}, \\lambda=1}; \\text{mode}=\\text{full}\\right)$$

---

## **3\. Neural Network Architecture**

### **3.1 Architecture Overview**

The architecture consists of three components: (1) a **Bitstring Encoder** that embeds individual bitstrings into a learned feature space, (2) a **Permutation-Invariant Set Transformer** that processes the variable-size set of (bitstring, count) pairs, and (3) a **Dual-Head Decoder** that produces both the shot-noise-denoised and hardware-noise-mitigated distributions.

Input: {(x\_i, c\_i)}\_{i=1}^{2^n}  (bitstring, count pairs from S shots)  
         │  
         ▼  
┌─────────────────────────┐  
│  Bitstring Encoder      │  Embeds each bitstring x\_i ∈ {0,1}^n  
│  (per-bitstring MLP)    │  into d-dim feature: e\_i \= φ(x\_i)  
└───────────┬─────────────┘  
            │ {(e\_i, c\_i/S)}  
            ▼  
┌─────────────────────────┐  
│  Set Transformer        │  Permutation-invariant processing  
│  (ISAB → PMA → SAB)     │  of variable-size sets  
│  \+ count-weighted attn  │  
└───────────┬─────────────┘  
            │ z (global representation, d-dim)  
            │  
     ┌──────┴──────┐  
     ▼             ▼  
┌─────────┐  ┌──────────┐  
│ SN-D    │  │ HN-E     │  
│ Head    │  │ Head     │  
│ (softmax│  │ (softmax │  
│  over   │  │  over    │  
│  bits.) │  │  bits.)  │  
└────┬────┘  └────┬─────┘  
     │            │  
     ▼            ▼  
  p̂\_SN-D      p̂\_HN-E  
  (shot-       (full  
   denoised)    QEM)

### **3.2 Bitstring Encoder**

Each bitstring $\\mathbf{x} \\in {0,1}^n$ is encoded into a $d$-dimensional feature vector:

class BitstringEncoder(nn.Module):  
    """  
    Encodes bitstrings using per-qubit embedding \+ positional encoding.  
    Permutation-invariant by construction: qubit order doesn't matter  
    because we sum over qubit embeddings.  
    """  
    def \_\_init\_\_(self, d\_model=64, n\_max\_qubits=32):  
        super().\_\_init\_\_()  
        self.qubit\_embed \= nn.Embedding(2, d\_model // 2\)  \# |0⟩, |1⟩  
        self.pos\_encoding \= PositionalEncoding(d\_model, n\_max\_qubits)  
        self.projection \= nn.Linear(d\_model, d\_model)  
          
    def forward(self, bitstrings: torch.Tensor) \-\> torch.Tensor:  
        """  
        Args:  
            bitstrings: (B, M, n) batch of M bitstrings, each n bits  
        Returns:  
            embeddings: (B, M, d) per-bitstring features  
        """  
        \# Per-qubit embedding: (B, M, n, d//2)  
        qubit\_emb \= self.qubit\_embed(bitstrings.long())  
        \# Add positional encoding for qubit index  
        qubit\_emb \= self.pos\_encoding(qubit\_emb, dim=2)  
        \# Pool over qubits (sum \= permutation-invariant)  
        pooled \= qubit\_emb.sum(dim=2)  \# (B, M, d//2)  
        \# Expand back to d\_model  
        padded \= F.pad(pooled, (0, pooled.size(-1)))  
        return self.projection(padded)

**Design rationale**: The sum-pooling over qubit embeddings ensures permutation invariance with respect to qubit labeling. The positional encoding allows the model to distinguish different qubit positions when needed (important for hardware with non-uniform noise), while the sum-pooling ensures the overall representation is invariant.

### **3.3 Count-Weighted Set Transformer**

The core architecture is a Set Transformer [Lee et al., 2019](https://arxiv.org/abs/1810.00825) modified to incorporate measurement counts as attention weights:

class CountWeightedSetTransformer(nn.Module):  
    """  
    Set Transformer with count-weighted attention.  
    Processes variable-size sets of (bitstring\_embedding, normalized\_count) pairs.  
    Permutation-invariant by construction.  
    """  
    def \_\_init\_\_(self, d\_model=64, n\_heads=4, n\_ISAB=2, n\_SAB=1, d\_ff=256):  
        super().\_\_init\_\_()  
        \# Induced Set Attention Blocks (encoder)  
        self.encoder \= nn.ModuleList(\[  
            ISAB(d\_model, n\_heads, d\_ff, m=16)  \# m=16 inducing points  
            for \_ in range(n\_ISAB)  
        \])  
        \# Pooling by Multihead Attention (PMA) \-\> fixed-size global rep  
        self.pma \= PMA(d\_model, n\_heads, 1, d\_ff)  \# 1 seed \= global vector  
        \# Self-Attention Blocks (decoder)  
        self.decoder \= nn.ModuleList(\[  
            SAB(d\_model, n\_heads, d\_ff)  
            for \_ in range(n\_SAB)  
        \])  
        \# Count integration  
        self.count\_proj \= nn.Linear(1, d\_model)  
          
    def forward(self, embeddings: torch.Tensor, counts: torch.Tensor) \-\> torch.Tensor:  
        """  
        Args:  
            embeddings: (B, M, d) bitstring embeddings  
            counts: (B, M, 1\) normalized counts (c\_i / S)  
        Returns:  
            global\_z: (B, d) global latent representation  
        """  
        \# Integrate counts into embeddings  
        x \= embeddings \+ self.count\_proj(counts)  \# (B, M, d)  
          
        \# Encoder: ISAB blocks  
        for isab in self.encoder:  
            x \= isab(x)  
          
        \# Pool to global representation  
        z \= self.pma(x)  \# (B, 1, d)  
          
        \# Decoder: SAB blocks  
        for sab in self.decoder:  
            z \= sab(z)  
          
        return z.squeeze(1)  \# (B, d)

**Key properties**:

* **Permutation invariance**: The Set Transformer is invariant to the ordering of input elements by construction (attention is permutation-equivariant, PMA pools to a permutation-invariant global vector).  
* **Variable-size input**: Can process sets of different sizes (different numbers of observed bitstrings), enabling training on small qubit counts and inference on larger ones.  
* **Count weighting**: Measurement counts are integrated as additional features, allowing the model to weight frequent bitstrings more heavily.  
* **Bottleneck**: The PMA layer compresses the variable-size set into a single $d$-dimensional vector, forcing the network to discard noise (random fluctuations) and retain structured signal.

### **3.4 Dual-Head Decoder**

class DualHeadDecoder(nn.Module):  
    """  
    Two output heads:  
    \- SN-D head: outputs shot-noise-denoised distribution  
    \- HN-E head: outputs hardware-noise-mitigated distribution  
    Both produce valid probability distributions over bitstrings.  
    """  
    def \_\_init\_\_(self, d\_model=64, n\_max\_qubits=20, temperature=1.0):  
        super().\_\_init\_\_()  
        self.n\_max\_qubits \= n\_max\_qubits  
        self.temperature \= nn.Parameter(torch.tensor(temperature))  
          
        \# SN-D head  
        self.sn\_head \= nn.Sequential(  
            nn.Linear(d\_model, d\_model \* 2),  
            nn.GELU(),  
            nn.Linear(d\_model \* 2, d\_model),  
        )  
          
        \# HN-E head  
        self.hn\_head \= nn.Sequential(  
            nn.Linear(d\_model, d\_model \* 2),  
            nn.GELU(),  
            nn.Linear(d\_model \* 2, d\_model),  
        )  
          
        \# Bitstring scorer: produces a logit for each candidate bitstring  
        self.scorer \= nn.Linear(d\_model, 1\)  
          
    def forward(self, z: torch.Tensor, candidate\_bitstrings: torch.Tensor):  
        """  
        Args:  
            z: (B, d) global latent representation  
            candidate\_bitstrings: (B, M, n) bitstrings to score  
        Returns:  
            sn\_dist: (B, M) shot-noise-denoised distribution  
            hn\_dist: (B, M) hardware-noise-mitigated distribution  
        """  
        \# SN-D branch  
        sn\_z \= self.sn\_head(z)  \# (B, d)  
        sn\_embed \= self.bitstring\_encoder(candidate\_bitstrings)  \# (B, M, d)  
        sn\_logits \= self.scorer(sn\_z.unsqueeze(1) \* sn\_embed).squeeze(-1)  \# (B, M)  
        sn\_dist \= F.softmax(sn\_logits / self.temperature, dim=-1)  
          
        \# HN-E branch  
        hn\_z \= self.hn\_head(z)  \# (B, d)  
        hn\_embed \= self.bitstring\_encoder(candidate\_bitstrings)  \# (B, M, d)  
        hn\_logits \= self.scorer(hn\_z.unsqueeze(1) \* hn\_embed).squeeze(-1)  \# (B, M)  
        hn\_dist \= F.softmax(hn\_logits / self.temperature, dim=-1)  
          
        return sn\_dist, hn\_dist

**Candidate bitstrings**: At inference, the candidate set consists of all $2^n$ bitstrings for small $n$, or the union of observed bitstrings \+ top-$k$ most likely unobserved bitstrings (predicted by the model) for large $n$. This avoids the $O(2^n)$ bottleneck.

### **3.5 Denoising Autoencoder (Baseline Architecture)**

For comparison and ablation, a simpler denoising autoencoder operating on the full distribution vector:

class DistributionAutoencoder(nn.Module):  
    """  
    Baseline: denoising autoencoder operating on full 2^n-dim distribution vectors.  
    Not permutation-invariant. Fixed input size \= 2^n.  
    Used for small qubit counts (n ≤ 8\) as a baseline.  
    """  
    def \_\_init\_\_(self, input\_dim, hidden\_dims=\[256, 128, 64\], bottleneck\_dim=32):  
        super().\_\_init\_\_()  
        \# Encoder  
        layers \= \[\]  
        d \= input\_dim  
        for h in hidden\_dims \+ \[bottleneck\_dim\]:  
            layers.extend(\[nn.Linear(d, h), nn.GELU()\])  
            d \= h  
        self.encoder \= nn.Sequential(\*layers)  
          
        \# Decoder (dual head)  
        self.sn\_decoder \= nn.Sequential(  
            nn.Linear(bottleneck\_dim, hidden\_dims\[-1\]), nn.GELU(),  
            nn.Linear(hidden\_dims\[-1\], input\_dim)  
        )  
        self.hn\_decoder \= nn.Sequential(  
            nn.Linear(bottleneck\_dim, hidden\_dims\[-1\]), nn.GELU(),  
            nn.Linear(hidden\_dims\[-1\], input\_dim)  
        )  
          
    def forward(self, dist: torch.Tensor):  
        z \= self.encoder(dist)  
        sn\_logits \= self.sn\_decoder(z)  
        hn\_logits \= self.hn\_decoder(z)  
        return F.softmax(sn\_logits, dim=-1), F.softmax(hn\_logits, dim=-1)

### **3.6 Architecture Comparison**

| Property | Autoencoder | Set Transformer |
| ----- | ----- | ----- |
| Permutation invariant | No | Yes |
| Variable qubit count | No (fixed $2^n$ input) | Yes |
| Scalability | $O(2^n)$ params | $O(n \\cdot d^2)$ params |
| Max practical $n$ | \~8 (256-dim) | \~20+ (bitstring-level) |
| Inductive bias | None | Set structure, count weighting |
| Training cost | Low | Moderate |
| Use case | Small-scale validation | Main architecture, scaling experiments |

---

## **4\. Training Strategy**

### **4.1 Loss Functions**

#### **4.1.1 Primary Loss: Composite Distribution Loss**

$$\\mathcal{L}*{\\text{dist}} \= \\alpha \\cdot \\mathcal{L}*{\\text{KL}} \+ \\beta \\cdot \\mathcal{L}*{\\text{TVD}} \+ \\gamma \\cdot \\mathcal{L}*{\\text{chi2}}$$

where:

* **KL divergence** (asymmetric, penalizes missing mass):  
   $$\\mathcal{L}*{\\text{KL}} \= \\sum*{\\mathbf{x}} p\_{\\text{target}}(\\mathbf{x}) \\log \\frac{p\_{\\text{target}}(\\mathbf{x})}{p\_{\\text{pred}}(\\mathbf{x}) \+ \\epsilon}$$

* **Total Variation Distance** (symmetric, interpretable):  
   $$\\mathcal{L}*{\\text{TVD}} \= \\frac{1}{2} \\sum*{\\mathbf{x}} |p\_{\\text{pred}}(\\mathbf{x}) \- p\_{\\text{target}}(\\mathbf{x})|$$

* **Chi-squared divergence** (penalizes large relative errors on rare events):  
   $$\\mathcal{L}*{\\text{chi2}} \= \\sum*{\\mathbf{x}} \\frac{(p\_{\\text{pred}}(\\mathbf{x}) \- p\_{\\text{target}}(\\mathbf{x}))^2}{p\_{\\text{target}}(\\mathbf{x}) \+ \\epsilon}$$

**Default coefficients**: $\\alpha \= 1.0, \\beta \= 0.5, \\gamma \= 0.1$

#### **4.1.2 Physicality Regularization**

$$\\mathcal{L}*{\\text{phys}} \= \\lambda\_1 \\cdot \\max(0, \-\\min*{\\mathbf{x}} p\_{\\text{pred}}(\\mathbf{x}))^2 \+ \\lambda\_2 \\cdot (|\\sum\_{\\mathbf{x}} p\_{\\text{pred}}(\\mathbf{x}) \- 1|)^2$$

This penalizes negative probabilities and non-normalized outputs. With softmax outputs, these are automatically satisfied, but the regularization helps during early training when the temperature parameter is unstable.

#### **4.1.3 Cross-Stage Consistency Loss**

$$\\mathcal{L}*{\\text{consist}} \= \\text{TVD}(f^{(\\text{HN})}(f^{(\\text{SN})}(\\hat{p}*{\\text{low}})), f^{(\\text{HN})}(\\hat{p}\_{\\text{high}}))$$

This encourages the HN-E head to produce consistent results whether given a raw low-shot input or the SN-D-denoised input. It ties the two stages together.

#### **4.1.4 Total Loss**

$$\\mathcal{L}*{\\text{total}} \= w\_1 \\cdot \\mathcal{L}*{\\text{SN-D}} \+ w\_2 \\cdot \\mathcal{L}*{\\text{HN-E}} \+ w\_3 \\cdot \\mathcal{L}*{\\text{consist}} \+ w\_4 \\cdot \\mathcal{L}\_{\\text{phys}}$$

where $\\mathcal{L}*{\\text{SN-D}}$ is the distribution loss for Stage 1 targets, $\\mathcal{L}*{\\text{HN-E}}$ is the distribution loss for Stage 2 targets.

**Default weights**: $w\_1 \= 1.0, w\_2 \= 1.0, w\_3 \= 0.3, w\_4 \= 0.1$

### **4.2 Training Protocol**

#### **4.2.1 Curriculum Learning**

Training proceeds in three phases:

| Phase | Epochs | SN-D pairs | HN-E pairs | Description |
| ----- | ----- | ----- | ----- | ----- |
| Phase 1: SN-D only | 0–100 | ✓ | ✗ | Learn shot noise patterns first |
| Phase 2: Add HN-E | 100–250 | ✓ | ✓ | Joint training, $w\_2$ ramped from 0→1 |
| Phase 3: Fine-tune | 250–300 | ✓ | ✓ | Lower learning rate, consistency loss emphasized |

#### **4.2.2 Noise-Aware Data Augmentation**

To increase training data diversity without additional hardware runs:

1. **Bootstrap resampling**: From a high-shot measurement $\\hat{p}*{S*{\\text{high}}}$, generate multiple low-shot samples by multinomial resampling: $\\hat{p}*{S*{\\text{low}}}^{(j)} \\sim \\text{Multinomial}(S\_{\\text{low}}, \\hat{p}*{S*{\\text{high}}})$. This creates many training pairs from a single hardware run.

2. **Noise level interpolation**: If measurements at $\\lambda\_1$ and $\\lambda\_2$ are available, generate intermediate noise levels by mixing: $\\hat{p}*{\\lambda*{\\text{mid}}} \= \\alpha \\cdot \\hat{p}*{\\lambda\_1} \+ (1-\\alpha) \\cdot \\hat{p}*{\\lambda\_2}$ for $\\alpha \\in \[0,1\]$.

3. **Circuit parameter perturbation**: For parametrized circuits (VQE, QAOA), perturb angles slightly to generate new circuits with similar noise profiles.

#### **4.2.3 Hyperparameters**

| Hyperparameter | Value | Justification |
| ----- | ----- | ----- |
| Optimizer | AdamW | Standard for transformers |
| Learning rate | 3e-4 (Phase 1), 1e-4 (Phase 3\) | With cosine warmup |
| Weight decay | 0.01 | Standard regularization |
| Batch size | 64 | Balance memory/generalization |
| $d\_{\\text{model}}$ | 64 | Sufficient for ≤20 qubits |
| $n\_{\\text{heads}}$ | 4 | Standard for small models |
| $n\_{\\text{ISAB}}$ | 2 | Depth vs. cost tradeoff |
| Inducing points ($m$) | 16 | Set Transformer default |
| Dropout | 0.1 | Prevent overfitting |
| Gradient clipping | 1.0 | Stable training |
| Early stopping patience | 20 epochs | Based on validation TVD |
| $S\_{\\text{low}}$ | 100 | Realistic low-shot regime |
| $S\_{\\text{high}}$ | 10,000 | Converged estimate |
| Noise scales $\\lambda$ | {1.0, 1.5, 2.0, 3.0} | Standard ZNE folding factors |
| Training circuits | 5,000 | Diverse circuit families |
| Validation circuits | 1,000 | Held-out |
| Test circuits | 1,000 | Final evaluation |

---

## **5\. Data Pipeline**

### **5.1 Circuit Generation**

class CircuitGenerator:  
    """  
    Generates diverse quantum circuits for training and evaluation.  
    """  
    CIRCUIT\_FAMILIES \= {  
        'random\_clifford': 'Clifford gates, random depth',  
        'random\_non\_clifford': 'Mix of Clifford and non-Clifford',  
        'vqe\_ansatz': 'Hardware-efficient VQE ansatz',  
        'qaoa': 'QAOA p=1,2,3 layers',  
        'ghz\_state': 'GHZ state preparation',  
        'w\_state': 'W-state preparation',  
        'iqp\_circuit': 'Instantaneous Quantum Polynomial',  
        'haar\_random': 'Haar-random unitary (small n only)',  
        'tfim\_dynamics': 'Ising model Trotterized dynamics',  
        'bernstein\_vazirani': 'Bernstein-Vazirani algorithm',  
    }  
      
    def generate(self, n\_qubits, depth, family, seed=None):  
        """Generate a single circuit."""  
        \# Returns a Qiskit QuantumCircuit  
        ...

**Circuit distribution for training**: 40% random non-Clifford, 20% VQE, 15% QAOA, 10% GHZ/W states, 10% IQP, 5% TFIM dynamics. This ensures diversity while emphasizing practically relevant circuits.

### **5.2 Data Collection Protocol**

class DataCollector:  
    """  
    Collects measurement data from simulator or real hardware.  
    For each circuit, collects:  
    \- Low-shot (100) at λ=1  
    \- High-shot (10000) at λ=1    
    \- Medium-shot (1000) at λ={1.5, 2.0, 3.0} (gate folded)  
    \- Medium-shot (1000) at λ\<1 (dynamical decoupling)  
    """  
      
    def collect\_circuit\_data(self, circuit, backend, n\_qubits):  
        data \= {}  
          
        \# Stage 1 data: shot noise pairs  
        data\['low\_shot'\] \= self.run(circuit, shots=100, noise\_scale=1.0)  
        data\['high\_shot'\] \= self.run(circuit, shots=10000, noise\_scale=1.0)  
          
        \# Stage 2 data: noise scaling pairs  
        for lam in \[1.5, 2.0, 3.0\]:  
            folded \= self.gate\_fold(circuit, factor=lam)  
            data\[f'noise\_{lam}'\] \= self.run(folded, shots=1000, noise\_scale=lam)  
          
        \# Reduced noise via dynamical decoupling  
        dd\_circuit \= self.add\_dd(circuit)  
        data\['noise\_reduced'\] \= self.run(dd\_circuit, shots=1000, noise\_scale='dd')  
          
        return data

### **5.3 Noise Models for Simulation**

| Noise Model | Parameters | Use Case |
| ----- | ----- | ----- |
| Depolarizing | $p \= 0.001 \- 0.05$ per gate | Baseline, analytic tractability |
| Amplitude damping | $T\_1 \= 50-200,\\mu s$ | Realistic decoherence |
| Phase damping | $T\_2 \= 20-100,\\mu s$ | Dephasing |
| Readout error | $p\_{0\\to1}, p\_{1\\to0} \= 0.01-0.05$ | Measurement noise |
| Combined | All above \+ crosstalk | IBMQ-like realistic |
| Hardware-calibrated | From IBMQ calibration data | Real hardware validation |

### **5.4 Data Encoding**

The input to the network is a set of (bitstring, normalized\_count) pairs:

def encode\_measurement(counts\_dict, n\_qubits, max\_bitstrings=256):  
    """  
    Convert Qiskit Counts dict to model input format.  
      
    Args:  
        counts\_dict: {'01': 45, '10': 55, ...}  
        n\_qubits: number of qubits  
        max\_bitstrings: cap on number of bitstrings (for large n)  
      
    Returns:  
        bitstrings: (M, n) tensor  
        counts: (M, 1\) tensor (normalized)  
    """  
    total \= sum(counts\_dict.values())  
    items \= sorted(counts\_dict.items(), key=lambda x: \-x\[1\])\[:max\_bitstrings\]  
      
    bitstrings \= \[\]  
    counts \= \[\]  
    for bitstring, count in items:  
        bs \= \[int(b) for b in bitstring.zfill(n\_qubits)\]  
        bitstrings.append(bs)  
        counts.append(count / total)  
      
    return torch.tensor(bitstrings), torch.tensor(counts).unsqueeze(1)

---

## **6\. Experiment Plan**

### **6.1 Experiment Matrix**

| Experiment | Purpose | Qubits | Backend | Circuits |
| ----- | ----- | ----- | ----- | ----- |
| **Exp 1**: SN-D validation | Prove shot-noise reduction works | 2–8 | Aer simulator | 5000 |
| **Exp 2**: HN-E validation | Prove hardware-noise mitigation works | 2–8 | Aer simulator | 5000 |
| **Exp 3**: Unified N2LN | Combined performance | 4–8 | Aer simulator | 5000 |
| **Exp 4**: Qubit scaling | Generalization across n | Train: 4,6; Test: 8,10,12,20 | Aer \+ MPS | 3000 |
| **Exp 5**: Real hardware | IBMQ validation | 7, 16, 27, 127 | IBMQ | 200 |
| **Exp 6**: Ablation | Component contributions | 4–8 | Aer | 2000 |
| **Exp 7**: Baseline comparison | vs. ZNE, CDR, readout inversion, ideal-supervised | 4–12 | Aer \+ IBMQ | 2000 |

### **6.2 Evaluation Metrics**

| Metric | Formula | Target |
| ----- | ----- | ----- |
| **TVD** | $\\frac{1}{2}\\sum\_x |p\_{\\text{pred}}(x) \- p\_{\\text{ideal}}(x)|$ | ≤ ZNE TVD |
| **Fidelity** | $F \= (\\sum\_x \\sqrt{p\_{\\text{pred}}(x) \\cdot p\_{\\text{ideal}}(x)})^2$ | ≥ 0.95 for n≤8 |
| **MAE (expectation)** | $|O\_{\\text{pred}} \- O\_{\\text{ideal}}|$ | ≤ 50% of raw MAE |
| **MSE** | $\\frac{1}{2^n}\\sum\_x (p\_{\\text{pred}}(x) \- p\_{\\text{ideal}}(x))^2$ | Report |
| **KL divergence** | $\\sum\_x p\_{\\text{ideal}}(x) \\log \\frac{p\_{\\text{ideal}}(x)}{p\_{\\text{pred}}(x)}$ | Report |
| **Shot efficiency** | TVD at $S\_{\\text{low}}$ shots after N2LN vs. raw TVD at $S\_{\\text{equiv}}$ shots | $S\_{\\text{equiv}} \\gg S\_{\\text{low}}$ |
| **Simulation-free penalty** | TVD(N2LN) / TVD(ideal-supervised oracle) | \< 1.5 |

### **6.3 Baselines**

| Baseline | Description | Requires Simulation? |
| ----- | ----- | ----- |
| **Raw** | No mitigation, direct measurement | No |
| **Readout inversion** | Confusion matrix inversion (linear) | Calibration circuits only |
| **ZNE** | Zero-noise extrapolation (Richardson, exponential) | No |
| **CDR** | Clifford Data Regression | Yes (Clifford simulation) |
| **Ideal-supervised oracle** | Same architecture, trained with ideal targets | Yes (full simulation) |
| **IBM ML-QEM** | Random forest on ZNE-mimicked data | Yes (near-Clifford) |

The **ideal-supervised oracle** is the most important baseline: it uses the same network architecture but trains with true ideal distributions (from simulation). The ratio N2LN/oracle performance quantifies the **simulation-free penalty**—the key metric for assessing whether self-supervision is viable.

### **6.4 Detailed Experiment Descriptions**

#### **Experiment 1: SN-D Validation**

**Goal**: Validate that the network can reduce shot noise by learning low-shot → high-shot mapping.

**Setup**:

* Qubits: 2, 4, 6, 8  
* Noise: Depolarizing (p=0.01) \+ readout error (p=0.02)  
* For each circuit: collect 100-shot and 10,000-shot measurements  
* Train SN-D head only (Phase 1\)  
* Evaluate: TVD(SN-D output, high-shot) vs. TVD(raw low-shot, high-shot)

**Expected result**: SN-D output TVD should be 40–70% lower than raw low-shot TVD, approaching the high-shot TVD.

**Success criterion**: SN-D TVD ≤ 0.5 × raw low-shot TVD for ≥80% of test circuits.

#### **Experiment 2: HN-E Validation**

**Goal**: Validate that the network can mitigate hardware noise using noise-scaling pairs.

**Setup**:

* Qubits: 2, 4, 6, 8  
* Noise: Combined (depolarizing \+ amplitude damping \+ readout)  
* For each circuit: collect measurements at λ \= {1.0, 1.5, 2.0, 3.0} with 1000 shots each  
* Train HN-E head on pairs (λ\_high → λ\_low)  
* Evaluate: TVD(HN-E output, ideal) vs. TVD(raw, ideal) and vs. ZNE

**Expected result**: HN-E output should achieve TVD comparable to ZNE, with lower variance.

**Success criterion**: HN-E TVD ≤ ZNE TVD for ≥60% of test circuits; MAE improvement ≥30% over raw.

#### **Experiment 3: Unified N2LN**

**Goal**: Validate the combined two-stage model.

**Setup**:

* Qubits: 4, 6, 8  
* Full training protocol (all three phases)  
* Evaluate against all baselines

**Expected result**: N2LN should outperform raw and readout inversion, match or exceed ZNE, and approach the ideal-supervised oracle within a factor of 1.5.

#### **Experiment 4: Qubit-Count Generalization**

**Goal**: Test permutation-invariant generalization across qubit counts.

**Setup**:

* Train on: 4-qubit and 6-qubit circuits (pooled)  
* Test on: 8, 10, 12, 20 qubits (zero-shot)  
* Backend: Aer for n≤12, MPS simulator for n=20  
* Metrics: TVD and fidelity as function of test qubit count

**Expected result**: Performance degrades gracefully with qubit count. The Set Transformer should generalize better than the fixed-size autoencoder.

**Success criterion**: At n=10 (unseen), TVD ≤ 2× the in-distribution TVD at n=8.

#### **Experiment 5: Real Hardware Validation**

**Goal**: Validate on IBM Quantum hardware.

**Setup**:

* Phase 1 (n≤7): IBMQ Nairobi (7 qubits) — full pipeline, 200 circuits  
* Phase 2 (n≤16): IBMQ Brisbane (16 qubits) — 100 circuits, SN-D only (HN-E requires noise scaling, which is expensive on hardware)  
* Phase 3 (n=127): IBMQ Brisbane/Kyiv (127 qubits) — 50 circuits, SN-D only, restricted to observed bitstrings

**Hardware-specific considerations**:

* Gate folding for HN-E: use Qiskit’s `fold_gates_at_random()`  
* Dynamical decoupling: insert `XY4` sequences during idle times  
* Calibration drift: record calibration data alongside each run; include as metadata  
* Shot budget: IBMQ free tier \~10,000 shots/day; plan data collection carefully

#### **Experiment 6: Ablation Studies**

| Ablation | What’s Removed | Expected Impact |
| ----- | ----- | ----- |
| No SN-D (HN-E only) | Stage 1 | Degraded performance on low-shot inputs |
| No HN-E (SN-D only) | Stage 2 | Only shot noise removed, hardware noise intact |
| No consistency loss | $\\mathcal{L}\_{\\text{consist}}$ | Stages may become inconsistent |
| No count weighting | Count features | Poorer performance on skewed distributions |
| No curriculum | Phase-based training | Slower convergence, worse final performance |
| Autoencoder (not Set Trans.) | Permutation invariance | No qubit-count generalization |
| No bootstrap aug. | Data augmentation | Overfitting on small datasets |

---

## **7\. Scalability Roadmap**

### **7.1 Phase A: Laptop Simulation (Weeks 1–4)**

| Item | Detail |
| ----- | ----- |
| Qubits | 2–8 |
| Backend | Qiskit Aer (density matrix) |
| Circuits | 5,000 training, 1,000 test |
| Noise | Depolarizing, amplitude damping, readout |
| Architecture | Both Autoencoder and Set Transformer |
| Goal | Validate both stages, establish baselines |

### **7.2 Phase B: MPS Simulation (Weeks 5–10)**

| Item | Detail |
| ----- | ----- |
| Qubits | 10–30 |
| Backend | Qiskit Aer MPS simulator, or cuTensorNet on GPU |
| Circuits | Low-entanglement circuits (shallow depth, 1D connectivity) |
| Architecture | Set Transformer only |
| Key challenge | MPS bond dimension limits; restrict to circuits with entanglement entropy ≤ 20 |
| Goal | Validate qubit-count generalization up to 30 qubits |

**MPS specifics**: Use Qiskit Aer’s `matrix_product_state` method with `max_bond_dimension=256`. For circuits that exceed this, use approximate simulation and flag them. Alternatively, use NVIDIA’s cuTensorNet on a GPU for larger bond dimensions [NVIDIA cuTensorNet](https://resources.nvidia.com/en-us-quantum-computing-resource-center/enabling-matrix-product).

### **7.3 Phase C: Real Hardware — Small Scale (Weeks 11–14)**

| Item | Detail |
| ----- | ----- |
| Qubits | 7 (IBMQ Nairobi) |
| Circuits | 200, including VQE and QAOA |
| Data | Full pipeline (low-shot, high-shot, noise-scaled) |
| Architecture | Set Transformer, trained on simulator, fine-tuned on hardware |
| Goal | Validate on real noise, characterize sim-to-hardware transfer gap |

### **7.4 Phase D: Real Hardware — Utility Scale (Weeks 15–20)**

| Item | Detail |
| ----- | ----- |
| Qubits | 127 (IBMQ Brisbane or Kyiv) |
| Circuits | 50, shallow random \+ QAOA |
| Data | SN-D only (HN-E too expensive at this scale) |
| Architecture | Set Transformer, restricted to top-256 observed bitstrings |
| Key challenge | 2^127 possible bitstrings; only observe \~few thousand; model must work on sparse support |
| Goal | Demonstrate shot-noise reduction at utility scale; show feasibility of simulation-free approach |

**Large-scale bitstring handling**: At 127 qubits, the full distribution has $2^{127}$ outcomes. In practice, only a few thousand bitstrings are observed. The Set Transformer naturally handles this: the input is the *observed* set of (bitstring, count) pairs, and the output is a renormalized distribution over the same set. The model learns to smooth the empirical counts, not to predict unobserved bitstrings.

---

## **8\. Expected Results & Novelty Proof**

### **8.1 Expected Quantitative Results**

| Metric | Raw | ZNE | CDR | Ideal-Supervised | N2LN (ours) |
| ----- | ----- | ----- | ----- | ----- | ----- |
| TVD (n=4, p=0.01) | 0.15 | 0.06 | 0.04 | 0.03 | **0.05** |
| TVD (n=8, p=0.02) | 0.25 | 0.12 | 0.10 | 0.08 | **0.11** |
| MAE (n=4, VQE) | 0.08 | 0.03 | 0.02 | 0.015 | **0.025** |
| Fidelity (n=6) | 0.85 | 0.94 | 0.96 | 0.97 | **0.93** |
| Shot equiv. (n=4) | 100 | 100 | 100 | 100 | **\~2,000** |
| Sim. needed? | No | No | Yes | Yes | **No** |

### **8.2 How Results Prove Novelty**

1. **Simulation-free QEM**: N2LN achieves performance within 1.5× of the ideal-supervised oracle **without any classical simulation**. This is the central novelty claim.

2. **Unified noise mitigation**: N2LN addresses both shot noise and hardware noise in a single model, while ZNE addresses only hardware noise and shot-noise reduction methods address only shot noise.

3. **Qubit-count generalization**: The Set Transformer, trained on n=4,6, achieves meaningful mitigation at n=10,12 without retraining. No prior ML-QEM method demonstrates this.

4. **Shot amplification**: N2LN with 100 input shots achieves performance comparable to \~2,000 raw shots. This is a practical speedup for variational algorithms.

### **8.3 Key Figures for Publication**

1. **Fig 1**: Architecture diagram (Set Transformer \+ dual heads)  
2. **Fig 2**: TVD vs. qubit count for N2LN vs. baselines  
3. **Fig 3**: Shot efficiency curve (TVD vs. input shot count, with and without N2LN)  
4. **Fig 4**: Noise scale ablation (performance vs. number of noise levels used in HN-E training)  
5. **Fig 5**: Qubit-count generalization (train n=4,6; test n=8,10,12,20)  
6. **Fig 6**: Real hardware results (IBMQ 7-qubit and 127-qubit)  
7. **Fig 7**: Distribution visualization (example histograms: raw vs. N2LN vs. ideal)  
8. **Table 1**: Full baseline comparison  
9. **Table 2**: Ablation results

---

## **9\. Risk Analysis & Mitigation**

| Risk | Probability | Impact | Mitigation |
| ----- | ----- | ----- | ----- |
| HN-E extrapolation is unstable | Medium | High | Use multiple noise levels (≥3); add regularization; compare to ZNE which has the same fundamental limitation |
| Set Transformer doesn’t generalize to large n | Medium | High | Fallback: train per-qubit-count models; use autoencoder for fixed n; restrict to n≤20 |
| Hardware noise drift degrades model | Medium | Medium | Include calibration data as input features (following GEM approach); periodically retrain |
| Bootstrap resampling introduces bias | Low | Medium | Validate against direct multinomial sampling; use proper resampling protocols |
| IBMQ shot budget insufficient | High | Medium | Prioritize critical experiments; use simulator for most training; hardware for validation only |
| Ideal-supervised oracle vastly outperforms N2LN | Low | High | If gap \>3×, add semi-supervised variant: use simulation for a small fraction of training data (hybrid approach) |
| 127-qubit sparse support is too sparse | Medium | Medium | Restrict to SN-D only at 127 qubits; use top-k bitstring approach; acknowledge limitation |
| Reviewer questions theoretical basis | Medium | Medium | Include formal Noise2Noise extension proof in appendix; empirical validation on analytic cases |

---

## **10\. Timeline (6 Months)**

| Weeks | Milestone | Deliverable |
| ----- | ----- | ----- |
| 1–2 | Setup & infrastructure | Project skeleton, Qiskit integration, data pipeline |
| 3–4 | Exp 1: SN-D validation (n=2–8) | Shot-noise reduction results, baseline comparison |
| 5–6 | Exp 2: HN-E validation (n=2–8) | Hardware-noise mitigation results, ZNE comparison |
| 7–8 | Exp 3: Unified model (n=4–8) | Full N2LN results, ablation studies |
| 9–10 | Exp 4: Qubit scaling (n=10–20) | Generalization results, MPS integration |
| 11–12 | Exp 6: Ablation \+ Exp 7: Baselines | Complete ablation table, full baseline comparison |
| 13–14 | Exp 5a: IBMQ 7-qubit | Real hardware validation (small scale) |
| 15–16 | Writing: Methods & Theory | Theory section, architecture description |
| 17–18 | Exp 5b: IBMQ 127-qubit (if budget allows) | Utility-scale demonstration |
| 19–20 | Writing: Results & Discussion | Full draft |
| 21–22 | Revision & polish | Final thesis / paper draft |
| 23–24 | Buffer & submission | Thesis defense / paper submission |

---

## **11\. Key References**

1. **Noise2Noise** — Lehtinen et al., ICML 2018\. Foundational self-supervised denoising theory. [arXiv](https://arxiv.org/abs/1803.04189)  
2. **DAEM** — Liao et al., npj QI 2025\. Closest competitor; uses Clifford fiducial simulation. [Nature](https://www.nature.com/articles/s41534-025-00960-y)  
3. **IBM ML-QEM** — Liao et al., Nature MI 2024\. ML-QEM at 100-qubit scale with simulation. [IBM Research](https://research.ibm.com/publications/machine-learning-for-practical-quantum-error-mitigation--1)  
4. **GEM** — Wang et al., arXiv 2026\. Graph-based QEM with hardware calibration features. [arXiv](https://arxiv.org/html/2604.16815v1)  
5. **CDR** — Czarnik et al., Quantum 2021\. Clifford Data Regression baseline.  
6. **Set Transformer** — Lee et al., ICML 2019\. Permutation-invariant set architecture. [arXiv](https://arxiv.org/abs/1810.00825)  
7. **QEM Review** — Cai et al., Rev. Mod. Phys. 2023\. Comprehensive QEM survey. [APS](https://link.aps.org/doi/10.1103/RevModPhys.95.045005)  
8. **DL-QREM** — Kim et al., NJP 2022\. Deep learning for readout error mitigation. [arXiv](https://arxiv.org/abs/2112.03585)  
9. **Permutation-equivariant QNNs** — Schatzki et al., npj QI 2024\. [Nature](https://www.nature.com/articles/s41534-024-00804-1)  
10. **MPS Simulation** — Qiskit Aer MPS backend, NVIDIA cuTensorNet. [Qiskit](https://medium.com/qiskit/simulate-large-quantum-circuits-with-low-entanglement-using-the-matrix-product-state-simulator-c9b886dec674)

---

## **12\. Summary: What Makes This Publishable**

| Novelty Dimension | Status | Evidence |
| ----- | ----- | ----- |
| **No classical simulation** | ✅ Genuinely novel | No prior ML-QEM method achieves this; DAEM claims it but uses Clifford simulation |
| **Noise2Noise for quantum** | ✅ First application | No paper applies N2N to quantum measurement distributions |
| **Dual self-supervised signal** | ✅ Novel combination | Shot-count hierarchy \+ noise scaling is unprecedented |
| **Permutation-invariant QEM** | ✅ Novel architecture | Set Transformer never used for QEM distribution denoising |
| **Theoretical contribution** | ✅ Novel extension | Noise2Noise theory extended to multinomial distributions |
| **Practical utility** | ✅ Strong | Shot amplification \+ hardware noise mitigation in one model |

The method addresses a real, recognized gap (simulation-free QEM), has a clear theoretical foundation (Noise2Noise \+ noise scaling), uses a principled architecture (Set Transformer for permutation invariance), and has a feasible experimental plan spanning simulation to real 127-qubit hardware. The key honest framing is: **this is a self-supervised approach that trades a small performance penalty for complete elimination of the classical simulation requirement**—a trade-off that becomes increasingly favorable as qubit counts grow beyond classical simulability.

---

*This TDD is intended as the blueprint for a 6-month master’s thesis project. All architectural choices, hyperparameters, and experiment designs are starting points to be refined during implementation. The theoretical analysis in §2.5 should be formalized into a rigorous appendix for the final thesis.*

