# Retrieval-Augmented Generation (RAG) for Large Language Models — Technical Notes
Source: “Retrieval-Augmented Generation for Large Language Models: A Survey” (arXiv:2312.10997).  
This document distills the core ideas, design choices, mathematics, and practical trade-offs for building RAG systems. It complements the local notes in [RAG/search_methods.md](./search_methods.md) by focusing on end-to-end system design and theory.

Contents
- 1. Executive Summary
- 2. RAG Pipeline at a Glance
- 3. Corpus Preparation, Chunking, and Metadata
- 4. Indexing and Approximate Nearest Neighbor (ANN)
- 5. Query Understanding and Retrieval Strategies
- 6. Reranking and Late Interaction
- 7. Fusion and Generation Mechanisms
- 8. Training Paradigms and Objectives
- 9. Evaluation Methodology
- 10. Practical System Design and Operations
- 11. Strengths, Limitations, and Trade-offs
- 12. Representative Methods and Where They Fit
- 13. Checklists and Recipes
- Appendix A. Useful Equations and Heuristics
- Appendix B. Open Challenges and Research Directions

---

## 1) Executive Summary
- Retrieval-Augmented Generation (RAG) injects external knowledge into LLM generation by retrieving relevant passages at inference time. This reduces hallucinations, improves factuality, and enables rapid domain adaptation without retraining the LLM on all new data.
- Core idea: factorize generation into retrieval p(d|x) and conditional generation p(y|x, d), then combine over top-k retrieved evidence (documents/passages). This separates dynamic knowledge (in index) from general linguistic competence (in model).
- Principal trade-offs:
  - Quality vs latency: more/better retrieval (k↑, stronger rerankers) improves grounding but increases end-to-end latency.
  - Precision vs coverage: narrow filters or aggressive rerankers reduce noise but risk missing evidence; broad recall increases noise and cognitive load on generator.
  - Retriever vs generator capacity: better retrievers reduce hallucinations; larger LLMs can be more robust to noisy contexts but also more prone to distraction if context is poorly curated.
  - Operational complexity: maintaining indexes, freshness, guardrails, privacy, and multi-tenant isolation can dominate engineering effort.

---

## 2) RAG Pipeline at a Glance
- Ingestion:
  - Normalize, segment into passages/chunks with overlap; extract metadata (title, source, timestamp, entity tags).
  - Create sparse and/or dense representations; build ANN/lexical indexes; schedule background refresh.
- Query understanding:
  - Turn-level parsing, conversation re-writing, domain routing, query expansion.
- Retrieval:
  - One or more retrievers (lexical, dense, hybrid) return top-k.
  - Optional multi-hop or iterative retrieval for compositional questions.
- Reranking and selection:
  - Late interaction (e.g., ColBERT) or cross-encoders (monoT5/CE-BERT); MMR/diversification; filter by metadata.
- Fusion and generation:
  - Early fusion (concatenate); Fusion-in-Decoder (FiD); per-layer retrieval (RETRO-style); retrieval-conditioned prompting with citations.
- Post-generation:
  - Attributions/citations, groundedness checks, refusal when unsupported, logging and telemetry.

---

## 3) Corpus Preparation, Chunking, and Metadata
- Chunking strategies:
  - Fixed windows (e.g., 512–1024 tokens) with overlap (e.g., 10–20%) to prevent boundary loss.
  - Semantic/recursive splitting (by headings, paragraphs, sentences) improves unit coherence but may fragment entities across chunks.
  - Multi-granularity indexing: index both coarse (section-level) and fine (paragraph/sentence-level) chunks to support hierarchical retrieval.
- Heuristics:
  - Prefer keeping title/headings and source in each chunk to aid rerankers and generation.
  - For tables/code, snapshot both raw and linearized text representations.
- Metadata design:
  - Include fields usable as hard filters (source, language, time ranges, tenant, access control).
  - Avoid over-filtering in early stages; prefer a cascade where filters tighten as needed.
- Embeddings:
  - Representations may blend content and metadata (e.g., [title] + [body]) or keep metadata for scoring priors only.

---

## 4) Indexing and Approximate Nearest Neighbor (ANN)
- Sparse vs dense retrieval:
  - Sparse lexical (e.g., BM25, SPLADE) excels at exact term matching and high precision for rare entities/keywords.
    - BM25 score:
      - score(d, q) = Σ_i IDF(q_i) · ((tf_i · (k1 + 1)) / (tf_i + k1 · (1 − b + b · |d|/avgdl)))
      - tf_i is term frequency in d; |d| is document length; avgdl is average doc length; k1 ∈ [1.2, 2.0]; b ≈ 0.75.
  - Dense bi-encoders learn semantic similarity; better recall for paraphrases and long-tail semantics.
    - Similarities: cosine sim(q, d) = (q · d) / (||q|| ||d||), or inner product. Cosine preferred with L2-normalization; inner product supports learned norms.
- ANN structures and trade-offs:
  - HNSW (graph-based):
    - Multi-layer small-world graph. Query uses greedy search with ef_search; build uses ef_construction.
    - Tuning: Increasing ef_search improves recall at higher latency; M (max neighbors) and ef_construction affect memory and build time.
    - Practical: High recall with sublinear time; memory overhead typically 16–64 bytes/vector for graph links (implementation-dependent).
  - IVF/Flat/PQ (vector quantization):
    - IVF partitions vectors into nlist coarse clusters by k-means; search probes nprobe clusters.
    - PQ compresses residuals into m subvectors coded with k=2^b codewords per subspace (b bits); memory drops from 4d bytes to (m · b)/8 bytes per vector (+ codebooks).
    - OPQ (rotation) reduces quantization error; ADC (asymmetric distance computation) re-ranks by distances to centroids/codebooks.
  - Hybrid retrieval:
    - Combine sparse and dense by:
      - Union then re-rank (score normalization via z-score, min-max, or softmax temperature).
      - Weighted linear fusion: S = α · S_sparse + (1 − α) · S_dense.
- Latency budgets:
  - CPU HNSW for mid-scale corpora; GPU FAISS for high QPS; cache warm top queries; isolate write-heavy and read-heavy nodes.

---

## 5) Query Understanding and Retrieval Strategies
- Conversational and multi-turn:
  - Query re-writing to canonical single-turn form; incorporate conversation state and user constraints.
- Query expansion:
  - RM3 pseudo-relevance feedback; doc2query/T5-based expansion; HyDE (hypothetical document embeddings) where the model drafts a pseudo-answer and embeds it.
- Multi-hop and decomposition:
  - Self-Ask/Decompose-and-Retrieve: break complex questions into sub-questions; breadth vs depth search.
- Diversity-promoting selection:
  - Maximal Marginal Relevance (MMR):
    - At step t, select d ∈ D \\ S maximizing: λ · sim(d, q) − (1 − λ) · max_{s∈S} sim(d, s)
    - λ ∈ [0, 1] trades relevance for diversity; sim is typically cosine/CE score.
- Negative mining:
  - In-batch negatives, BM25 hard negatives, cross-encoder mined negatives; curriculum from easy → hard to stabilize training.

---

## 6) Reranking and Late Interaction
- Cross-encoders:
  - Jointly encode [q ; d] and predict relevance. Strong quality, O(k) forward passes amortized at higher latency.
  - Training objectives: pointwise cross-entropy, pairwise hinge or logistic for preference ordering.
- Late interaction (e.g., ColBERT):
  - Token-level embeddings for q and d; MaxSim aggregation:
    - score(q, d) = Σ_{i ∈ tokens(q)} max_{j ∈ tokens(d)} cos(q_i, d_j)
  - Trades between cross-encoder quality and bi-encoder speed; supports efficient ANN with special indexes.
- Cascades:
  - Stage 1: fast bi-encoder (N → k0 ≈ 200–1000).
  - Stage 2: late interaction (k0 → k1 ≈ 50–200).
  - Stage 3: cross-encoder (k1 → k_final ≈ 5–20).
  - Early exits: stop if score gap exceeds threshold; adapt k to latency budget.

---

## 7) Fusion and Generation Mechanisms
- Early fusion (concatenate passages):
  - Simple; bounded by context window; susceptible to noise/order effects.
- Fusion-in-Decoder (FiD):
  - Encode each passage independently; decoder attends to the concatenation of encoded representations.
  - Scales better with k than early fusion; memory ∝ k · encoder activations.
- Per-layer or per-step retrieval (RETRO-like):
  - Retrieve nearest neighbors for fixed-size text chunks during decoding; reduces reliance on long context windows.
  - Costs: frequent retrieval calls; careful batching and caching required.
- Prompt formatting and ordering:
  - Order passages by final score or interleave by source/time; include titles and source ids for attributions.
  - Use structured templates to encourage citations and abstention when evidence is missing.
- Guardrails:
  - Instructional constraints for groundedness and refusal; mention of uncertainty thresholds.

---

## 8) Training Paradigms and Objectives
- Retriever pretraining (dual-encoder contrastive):
  - InfoNCE/softmax contrastive loss:
    - L = −Σ_{(q, d⁺)} log ( exp(sim(q, d⁺)/τ) / Σ_{d ∈ {d⁺ ∪ N(q)}} exp(sim(q, d)/τ) )
    - τ is temperature; N(q) are negatives (in-batch or mined).
  - Triplet/margin losses:
    - L = max(0, m − sim(q, d⁺) + sim(q, d⁻))
- Distillation:
  - Cross-encoder teacher → bi-encoder student via KL on scores or pairwise ordering losses; improves semantic precision.
- End-to-end RAG training:
  - Marginal likelihood over retrieved docs:
    - p(y|x) = Σ_{d ∈ D} p(y|x, d; θ_g) · p(d|x; θ_r)
    - Optimize θ = {θ_g, θ_r} via max log p(y|x) with top-k approximation; optionally stop-gradient through retrieval to stabilize.
  - Alternatives:
    - RePlug-style: update retriever to increase the generator likelihood on correct outputs, using generator feedback as weak supervision.
- Reranker training:
  - Pointwise/pairwise with human or heuristic labels (MS MARCO, NQ); calibrate probabilities for thresholding.
- RL variants:
  - Retrieval policy optimization using downstream rewards (answer F1, groundedness score); off-policy reweighting to keep training stable.

---

## 9) Evaluation Methodology
- Tasks:
  - Open-domain QA (NQ, TriviaQA), fact checking (FEVER), long-form QA (ASQA/ELI5), KILT tasks, knowledge-grounded generation.
- Generation metrics:
  - Exact Match (EM), token-level F1, Rouge-L, BLEU.
  - Long-form: support coverage and faithfulness; answer aspect coverage measures.
  - Calibration: AUC of refusal vs correctness; ECE for probability calibration.
- Retrieval metrics:
  - Recall@k: fraction of queries with at least one gold evidence in top k.
  - MRR: (1/|Q|) · Σ_q 1/rank_q (rank of first relevant document).
  - nDCG@k: DCG/IDCG with gains from graded relevance; DCG@k = Σ_{i=1..k} (2^{rel_i} − 1)/log2(i+1).
  - Oracle upper bounds: FiD-oracle (generator given gold evidence) to measure headroom beyond retrieval.
- Faithfulness/groundedness:
  - Entailment-based checks between claims and retrieved passages; attribution precision/recall.
  - RAGAS-style metrics combining answer correctness, attribution correctness, and faithfulness indicators.
- Efficiency:
  - P50/P95 latency, QPS under concurrency, index build time, memory footprint; ablate per-stage timings to find bottlenecks.

---

## 10) Practical System Design and Operations
- Caching:
  - Cache query embeddings, top-k candidates, and reranker outputs; align cache TTL with index freshness SLA.
  - For conversational sessions, cache turn-local context vectors and partial fusion encodings.
- Sharding and scaling:
  - Document sharding (hash or semantic); maintain shard-local HNSW graphs to keep recall stable.
  - Consistent hashing for multi-tenant isolation; hot-shard mitigation via replication.
- Online updates and freshness:
  - Staging shadow indexes; atomic pointer swap after build; background HNSW insertion; consistency checks and canaries.
- Observability:
  - Trace per-stage scores, latencies, and doc ids; analyze drop-offs (recall vs rerank vs generation).
  - Quality guardrails: blocklisted sources, source diversity constraints, minimum evidence score thresholds.
- Safety and privacy:
  - Prompt-injection robust retrieval (sanitize and sandbox retrieved HTML/JS); source whitelisting; PII detection and redaction.
  - Tenant data isolation; access control enforced in retrieval layer.
- Cost:
  - GPU vs CPU trade-offs; small retriever + strong reranker; hybrid sparse-first cascades; knowledge distillation of rerankers.

---

## 11) Strengths, Limitations, and Trade-offs
- Strengths:
  - Rapid knowledge updates without re-training; improved factuality; domain adaptation via corpora swap.
- Limitations:
  - Domain-mismatch between embeddings and corpus; long-tail entities; noisy or redundant retrieved passages; distraction and “off-by-one” context errors.
- Trade-offs:
  - k and reranking depth increase quality but hurt latency; hybrid retrieval reduces misses but complicates calibration.
  - Scaling laws: diminishing returns as k grows; quality gains often come more from better retrieval/reranking than from increasing generator size beyond a point.

---

## 12) Representative Methods and Where They Fit
- REALM (Guu et al.): joint retrieval + LM pretraining with latent variable docs.
- DPR (Karpukhin et al.): dual-encoder retriever trained with in-batch negatives.
- RAG (Lewis et al.): marginalize generator over retrieved docs; end-to-end or two-stage.
- FiD (Izacard & Grave): encode passages separately; decoder attends across all encodings.
- RETRO (Borgeaud et al.): per-chunk retrieval during generation with external memory index.
- Atlas (Izacard et al.): retrieval-augmented fine-tuning for strong downstream QA.
- RePlug (Shi et al.): train retriever using generator feedback without end-to-end backprop.
- Contriever: unsupervised contrastive retriever; strong zero-shot dense retrieval.
- ColBERT(v2): late interaction with MaxSim and efficient indexing.
- SPLADE: sparse learned term expansions using regularization; bridges lexical-semantic gap.
- HyDE: hypothetical document expansion via LLM-generated pseudo-answers.
- SELF-ASK / decomposition methods: multi-hop reasoning via question decomposition.
- kNN-LM: datastore nearest neighbors to next-token predictions; complements RAG for local fluency/memory.

---

## 13) Checklists and Recipes
- Minimal Viable RAG (MVR):
  - Ingest: sentence/paragraph chunks (512–768 tokens), 10–20% overlap; metadata: title, url, date, tags.
  - Index: Hybrid (BM25 + HNSW dense); L2-normalized embeddings; k0=200.
  - Rerank: Cross-encoder to top-20; optional ColBERT stage if latency allows.
  - Fusion: FiD-style prompt with ordered passages by rerank score; include citations.
  - Metrics: Recall@k, MRR, EM/F1; groundedness checks; P50/P95 latency.
- Production hardening:
  - Hard-negative mining refresh; HyDE-based query expansion for recall; MMR diversification to reduce redundancy.
  - Caching at all stages; freshness via shadow index; observability dashboards; safety filters.
- Triage playbook:
  - Low recall: tune k0, nprobe/ef_search, hybrid weights; add HyDE; improve negatives; adjust chunk size/overlap.
  - Noisy context: tighten reranker, add MMR/diversity, better chunking, metadata filters.
  - Hallucinations: increase evidence threshold stronger grounding prompts; teach abstention; penalize unsupported claims.
  - Latency regressions: reduce k, remove stages, increase cache hit rate, route heavy queries to GPU FAISS.

---

## Appendix A. Useful Equations and Heuristics
- BM25:
  - score(d, q) = Σ_i IDF(q_i) · ((tf_i · (k1 + 1)) / (tf_i + k1 · (1 − b + b · |d|/avgdl)))
- Cosine similarity:
  - cos(u, v) = (u · v) / (||u|| ||v||); use L2-normalization and temperature scaling for stable softmaxes.
- InfoNCE (contrastive):
  - L = −Σ_{(q, d⁺)} log ( exp(sim(q, d⁺)/τ) / Σ_{d ∈ {d⁺ ∪ N(q)}} exp(sim(q, d)/τ) )
- Triplet/margin:
  - L = max(0, m − sim(q, d⁺) + sim(q, d⁻))
- MMR selection:
  - argmax_{d ∈ D \\ S} [ λ · sim(d, q) − (1 − λ) · max_{s ∈ S} sim(d, s) ]
- RAG marginal likelihood:
  - p(y|x) = Σ_{d ∈ D} p(y|x, d; θ_g) · p(d|x; θ_r), trained by maximizing log p(y|x) with top-k approximation.
- nDCG@k:
  - DCG@k = Σ_{i=1..k} (2^{rel_i} − 1)/log2(i + 1); nDCG@k = DCG@k / IDCG@k.
- Practical defaults (starting points; tune empirically):
  - Chunk length: 512–768 tokens; overlap 10–20%; k0=200 (retriever), k1=50 (late interaction), k_final=10–20 (CE).
  - HNSW: M ∈ [16, 48], ef_construction ∈ [100,400], ef_search ∈ [64, 256].
  - IVF-PQ: nlist ≈ 4√N; nprobe ∈ [1, 32]; m · b tuned to balance memory and recall.

---

## Appendix B. Open Challenges and Research Directions
- Long-context vs retrieval:
  - Larger context windows reduce the need for retrieval in some cases, but retrieval remains critical for freshness, efficiency, and grounding across large corpora.
- Dynamic retrieval:
  - Adaptive k and routing based on uncertainty; retrieval policies conditioned on question type and difficulty.
- Faithfulness:
  - Attribution-aware decoding; retrieval-aware loss functions; disentangling relevance vs support.
- Robustness:
  - Domain shift from general embeddings; adversarial/doc noise; prompt injection via retrieved content.
- Memory + RAG hybrids:
  - kNN-LM + RAG; differentiable memories; life-long learning via combined caches and indexes.
- Data governance:
  - Copyright and licensing compliance; user privacy; auditability of sources and decisions.

---

Notes
- This is an implementation-agnostic summary designed for practitioners and researchers. It focuses on the theoretical and systems aspects (formulas, trade-offs, and design patterns). No code is included by design.
- For taxonomy and additional search techniques, see [RAG/search_methods.md](./search_methods.md).
