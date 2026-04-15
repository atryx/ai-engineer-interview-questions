<!-- GitAds-Verify: AHYZ3QO36YNN1LCNGH8G4HR1E7WXYUUE -->

# AI Engineer Interview Questions

[![Sponsored by GitAds](https://gitads.dev/v1/ad-serve?source=atryx/ai-engineer-interview-questions@github)](https://gitads.dev/v1/ad-track?source=atryx/ai-engineer-interview-questions@github)

> 100+ AI engineer interview questions with detailed answers — LLMs, RAG, agents, embeddings, evaluation, fine-tuning, and production AI systems. The questions companies are actually asking in 2026.

**Not textbook ML theory. Real AI engineering interviews.**

---

## Table of Contents

- [LLM Fundamentals](#llm-fundamentals)
- [Prompt Engineering](#prompt-engineering)
- [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
- [Embeddings & Vector Databases](#embeddings--vector-databases)
- [AI Agents & Tool Use](#ai-agents--tool-use)
- [Fine-Tuning & Training](#fine-tuning--training)
- [Evaluation & Testing](#evaluation--testing)
- [Production & Infrastructure](#production--infrastructure)
- [Safety, Ethics & Guardrails](#safety-ethics--guardrails)
- [System Design for AI](#system-design-for-ai)
- [Scenario-Based Questions](#scenario-based-questions)
- [Related Repos](#related-repos)
- [Contributing](#contributing)

---

## LLM Fundamentals

### Explain the Transformer architecture at a high level. Why did it replace RNNs?
**Level:** Junior

The Transformer processes entire sequences in parallel using **self-attention**, unlike RNNs which process tokens sequentially. Key components:

1. **Self-attention:** Each token attends to every other token, computing relevance weights. Captures long-range dependencies.
2. **Multi-head attention:** Multiple attention heads capture different relationship types (syntax, semantics, position).
3. **Feed-forward layers:** Process each position independently after attention.
4. **Positional encoding:** Since there's no recurrence, position information is added explicitly.

Why it replaced RNNs:
- **Parallelizable:** Process all tokens simultaneously (massive GPU speedup)
- **Long-range dependencies:** Direct attention between any two tokens (no vanishing gradient)
- **Scalable:** Scales to billions of parameters (GPT-4, Claude, Gemini)

### What is the difference between encoder-only, decoder-only, and encoder-decoder models?
**Level:** Mid

| Architecture | Examples | Use Case |
|-------------|----------|----------|
| **Encoder-only** | BERT, RoBERTa | Classification, NER, embeddings (understanding) |
| **Decoder-only** | GPT-4, Claude, Llama | Text generation, chat, code (generation) |
| **Encoder-decoder** | T5, BART | Translation, summarization (sequence-to-sequence) |

In 2026, **decoder-only** dominates for general-purpose AI engineering. Encoder models are still used for embeddings and classification.

### What are the key parameters when calling an LLM API? What does each one do?
**Level:** Junior

| Parameter | What It Does | Typical Range |
|-----------|-------------|---------------|
| **temperature** | Controls randomness. 0 = deterministic, 1 = creative | 0 - 2.0 |
| **top_p** (nucleus sampling) | Only sample from top cumulative probability tokens | 0.1 - 1.0 |
| **max_tokens** | Maximum output length | Depends on context window |
| **stop sequences** | Strings that stop generation | Task-specific |
| **frequency_penalty** | Penalize tokens that appear often (reduce repetition) | 0 - 2.0 |
| **presence_penalty** | Penalize tokens that appeared at all (encourage diversity) | 0 - 2.0 |

**Key insight:** Temperature and top_p both control randomness — use one or the other, not both. For factual tasks, use temperature=0. For creative tasks, temperature=0.7-1.0.

### Explain the difference between context window and knowledge cutoff.
**Level:** Junior

- **Context window:** Maximum number of tokens the model can process in a single request (input + output). GPT-4o: 128K, Claude: 200K, Gemini: 2M. Larger window ≠ better performance on all tasks.
- **Knowledge cutoff:** Last date of training data. The model doesn't know about events after this date. RAG solves this by providing current information in the context.

### What is tokenization and why does it matter for AI engineers?
**Level:** Mid

Tokenization splits text into subword units (tokens). Different models use different tokenizers (BPE, SentencePiece, WordPiece).

Why it matters:
- **Cost:** API pricing is per-token. Efficient prompts = lower cost.
- **Context limits:** Long inputs consume tokens, leaving less room for output.
- **Non-English text:** Some languages require more tokens per word (2-3× more expensive).
- **Code:** Whitespace and syntax tokens add up quickly.
- **Counting:** "chatGPT" might be 1-3 tokens depending on the tokenizer. Never assume 1 word = 1 token.

Rule of thumb: ~4 characters ≈ 1 token for English text.

---

## Prompt Engineering

### What are the main prompting strategies and when do you use each?
**Level:** Mid

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Zero-shot** | No examples, just instructions | Simple tasks, powerful models |
| **Few-shot** | 2-5 examples in the prompt | When format/style matters |
| **Chain-of-thought (CoT)** | "Think step by step" | Math, logic, complex reasoning |
| **ReAct** | Reason + Act — interleave thinking with tool use | Agent tasks with external tools |
| **Self-consistency** | Generate multiple answers, take majority vote | When correctness is critical |
| **Tree-of-thought** | Explore multiple reasoning paths | Complex problem-solving |

### How do you structure a system prompt for production use?
**Level:** Mid

A production system prompt should have:

```
1. Role definition: "You are a customer support agent for Acme Corp."
2. Behavioral constraints: "Never reveal internal pricing. Always be polite."
3. Output format: "Respond in JSON with fields: answer, confidence, sources."
4. Guardrails: "If you don't know, say 'I'll escalate this to a human agent.'"
5. Examples: 1-2 examples of ideal responses.
```

**Key principles:**
- Put the most important instructions first (primacy effect)
- Be explicit about what NOT to do
- Test adversarially (prompt injection attempts)
- Version control your prompts (they're code)

### How do you prevent prompt injection?
**Level:** Senior

Prompt injection = user input manipulating the system prompt. Defense-in-depth:

1. **Input sanitization:** Strip known attack patterns, limit input length
2. **Delimiter isolation:** Clearly separate system prompt from user input with delimiters
3. **Output validation:** Check LLM output against expected format/schema before acting
4. **Least privilege:** LLM tools should have minimum required permissions
5. **Dual-LLM pattern:** One LLM generates, another validates/filters
6. **Instruction hierarchy:** Use model features that prioritize system prompt over user input
7. **Canary tokens:** Include secret tokens in system prompt; if they appear in output, injection detected

**No single defense is foolproof.** Layer multiple defenses.

---

## RAG (Retrieval-Augmented Generation)

### What is RAG and why is it the most common production AI pattern?
**Level:** Junior

RAG combines retrieval (search) with generation (LLM). Instead of relying on the model's training data:

1. **Retrieve:** Search a knowledge base for relevant documents
2. **Augment:** Insert retrieved documents into the LLM's context
3. **Generate:** LLM generates an answer grounded in the retrieved content

Why it dominates:
- **Current information:** No retraining needed — update the knowledge base
- **Reduced hallucination:** Model cites actual documents
- **Domain-specific:** Works with proprietary/internal data
- **Cost-effective:** Cheaper than fine-tuning for most use cases
- **Auditable:** You can trace answers back to source documents

### Walk through a production RAG pipeline.
**Level:** Mid

```
Documents → Chunking → Embedding → Vector DB (indexing)
                                         ↓
User Query → Embed Query → Similarity Search → Top-K Chunks
                                                    ↓
                                         Reranker (optional)
                                                    ↓
                                    Prompt: System + Chunks + Query → LLM → Answer
```

Key decisions at each stage:
- **Chunking:** Fixed-size (512 tokens), semantic (paragraph boundaries), or recursive (split by headers → paragraphs → sentences)
- **Embedding model:** OpenAI `text-embedding-3-large`, Cohere `embed-v4`, or open-source (BGE, E5)
- **Vector DB:** Pinecone, Weaviate, Qdrant, pgvector, Chroma
- **Similarity metric:** Cosine similarity (most common), dot product, Euclidean
- **Top-K:** Usually 3-10 chunks. More isn't always better (noise dilutes signal)
- **Reranker:** Cross-encoder that reranks initial results for better relevance (Cohere Rerank, BGE-reranker)

### What are the common failure modes of RAG and how do you fix them?
**Level:** Senior

| Failure Mode | Symptom | Fix |
|-------------|---------|-----|
| **Bad chunking** | Relevant info split across chunks | Overlap chunks (20%), use semantic chunking |
| **Embedding mismatch** | Query finds wrong documents | Fine-tune embedding model on domain data, use hybrid search |
| **Missing context** | Answer is correct but incomplete | Increase top-K, add metadata filtering, use parent-document retriever |
| **Hallucination despite context** | LLM ignores retrieved docs | Stronger system prompt ("Only use provided context"), lower temperature |
| **Outdated index** | Answers reference old data | Incremental indexing pipeline, freshness metadata |
| **Lost in the middle** | Model ignores middle chunks | Put most relevant chunks first and last (primacy/recency bias) |

### What is hybrid search and when should you use it?
**Level:** Mid

Hybrid search combines **vector search** (semantic similarity) with **keyword search** (BM25/TF-IDF). Why:

- Vector search is great for semantic meaning ("how do I deploy?") but misses exact matches ("error code E4532")
- Keyword search catches exact terms but misses semantic similarity
- Combine with Reciprocal Rank Fusion (RRF) or weighted scoring

```python
# Pseudocode for hybrid search
vector_results = vector_db.search(embed(query), top_k=20)
keyword_results = bm25_index.search(query, top_k=20)
final_results = reciprocal_rank_fusion(vector_results, keyword_results)
return final_results[:10]
```

**Use hybrid search when:** Your data has specific identifiers (error codes, product names, SKUs) or technical terms that must match exactly.

---

## Embeddings & Vector Databases

### How do embedding models work and what makes a good embedding?
**Level:** Mid

Embedding models convert text into dense numerical vectors (768-3072 dimensions) where semantic similarity corresponds to vector proximity.

Training: Contrastive learning — train on (query, relevant_document) pairs. Model learns to place similar texts close together in vector space.

What makes a good embedding:
- **Captures semantic meaning** (not just word overlap)
- **Dimensionality:** Higher dimensions = more expressive but more expensive to search
- **Instruction-tuned:** Models like E5 accept task prefixes ("query:", "passage:") for better retrieval
- **Domain-appropriate:** General embeddings may underperform on specialized domains (medical, legal)

### Compare vector database options for production.
**Level:** Senior

| Database | Type | Strengths | Weaknesses |
|----------|------|-----------|------------|
| **Pinecone** | Managed SaaS | Zero-ops, scales well, fast | Vendor lock-in, cost at scale |
| **Weaviate** | Self-hosted/Cloud | Hybrid search built-in, modules | Operational complexity |
| **Qdrant** | Self-hosted/Cloud | Fast, Rust-based, rich filtering | Smaller community |
| **pgvector** | PostgreSQL extension | Use existing Postgres, no new infra | Slower at scale, limited features |
| **Chroma** | Embedded | Easy development, in-process | Not for production scale |
| **Milvus** | Self-hosted | Handles billions of vectors | Complex deployment |

**For most teams starting out:** pgvector (if already using Postgres) or Pinecone (if you want managed).

### What is the curse of dimensionality in vector search?
**Level:** Senior

As dimensions increase, all vectors become approximately equidistant — making similarity search less meaningful. Practical implications:

- Above ~1500 dimensions, exact search becomes very expensive
- Approximate Nearest Neighbor (ANN) algorithms are required: HNSW, IVF, ScaNN
- HNSW (Hierarchical Navigable Small World) is the most popular: builds a multi-layer graph for O(log n) search
- Trade-off: recall (accuracy) vs latency. Tune `ef_construction` and `ef_search` parameters.

---

## AI Agents & Tool Use

### What is an AI agent and how does it differ from a simple LLM chain?
**Level:** Mid

| Feature | LLM Chain | AI Agent |
|---------|-----------|----------|
| Flow | Predefined sequence of steps | Dynamic — LLM decides next action |
| Tools | None or static | Can call APIs, search, execute code |
| Planning | No planning | Plans, executes, observes, re-plans |
| State | Stateless between steps | Maintains state across actions |
| Autonomy | Low | High |

Agent loop:
```
while not done:
  action = llm.decide(observation, tools, history)
  observation = execute(action)
  if action == "final_answer":
    done = True
```

### What are the main agent architectures?
**Level:** Senior

| Architecture | How It Works | Best For |
|-------------|-------------|----------|
| **ReAct** | Reason → Act → Observe loop | Simple tool-use tasks |
| **Plan-and-Execute** | Create full plan first, then execute steps | Multi-step tasks with clear goals |
| **Reflection** | Agent critiques its own output and iterates | Code generation, writing |
| **Multi-agent** | Multiple specialized agents collaborate | Complex workflows (researcher + coder + reviewer) |
| **Hierarchical** | Manager agent delegates to worker agents | Enterprise orchestration |

### How do you make agents reliable in production?
**Level:** Senior

Agents are unreliable by default. Production hardening:

1. **Constrain the action space:** Only expose tools the agent needs. Don't give a support bot access to delete accounts.
2. **Structured output:** Force tool calls into JSON schema (function calling) instead of free-text parsing.
3. **Timeout and retry:** Set max iterations (10-20), timeout per step, exponential backoff on failures.
4. **Human-in-the-loop:** Require approval for high-stakes actions (payments, deletions, emails).
5. **Observability:** Log every step — input, reasoning, tool call, output. You'll need it for debugging.
6. **Fallback:** If agent fails after max retries, escalate to human or return a safe default.
7. **Cost guardrails:** Set max tokens/cost per agent run. A loop bug can burn through API credits fast.

---

## Fine-Tuning & Training

### When should you fine-tune vs use RAG vs prompt engineering?
**Level:** Senior

| Approach | When to Use | Cost | Effort |
|----------|-------------|------|--------|
| **Prompt engineering** | Always start here. Sufficient for most tasks. | Low | Hours |
| **RAG** | Need current/proprietary knowledge. Factual Q&A. | Medium | Days |
| **Fine-tuning** | Need specific style/format, domain adaptation, or cost reduction at scale. | High | Weeks |
| **RAG + Fine-tuning** | Maximum quality. Fine-tune for style, RAG for knowledge. | Highest | Weeks |

**Decision framework:**
- Can you solve it with a better prompt? → Don't fine-tune.
- Is the knowledge in your data? → RAG first.
- Do you need the model to behave differently (format, tone, specialized reasoning)? → Fine-tune.
- Are you spending $10K+/month on API calls? → Fine-tune a smaller model to reduce cost.

### Explain the fine-tuning process for LLMs.
**Level:** Mid

1. **Prepare data:** Create (instruction, response) pairs. Quality > quantity. 100-1000 high-quality examples.
2. **Choose base model:** Open-source (Llama 3, Mistral, Qwen) or provider (OpenAI, Claude fine-tuning).
3. **Choose method:**
   - **Full fine-tuning:** Update all parameters. Requires significant GPU (expensive).
   - **LoRA (Low-Rank Adaptation):** Only update small adapter matrices. 10-100× cheaper. Most common.
   - **QLoRA:** LoRA on quantized model. Fine-tune 70B models on a single GPU.
4. **Train:** Typically 1-5 epochs. Watch for overfitting (val loss increasing).
5. **Evaluate:** Automated metrics + human evaluation. Compare against base model.
6. **Deploy:** Merge LoRA weights or serve with adapter.

### What is RLHF and why does it matter?
**Level:** Mid

**Reinforcement Learning from Human Feedback** — the technique that made ChatGPT useful:

1. **SFT (Supervised Fine-Tuning):** Train on human-written ideal responses
2. **Reward Model:** Train a model to predict human preferences (which response is better?)
3. **PPO/DPO:** Optimize the LLM to maximize the reward model's score

**DPO (Direct Preference Optimization)** is replacing PPO — simpler, no separate reward model needed. Directly trains on preference pairs (chosen vs rejected responses).

Why it matters: Base models can be factual but unhelpful. RLHF makes them helpful, harmless, and honest.

---

## Evaluation & Testing

### How do you evaluate an LLM-powered application?
**Level:** Senior

There is no single metric. Use a layered approach:

**Level 1 — Component evaluation:**
- **Retrieval quality (RAG):** Precision@K, Recall@K, MRR, NDCG
- **Generation quality:** BLEU, ROUGE (weak proxies), BERTScore (semantic similarity)

**Level 2 — LLM-as-judge:**
- Use a strong model (GPT-4, Claude) to evaluate outputs on criteria:
  - Relevance (1-5): Does the answer address the question?
  - Faithfulness (1-5): Is the answer grounded in provided context?
  - Completeness (1-5): Are all aspects of the question covered?
  - Harmlessness (pass/fail): No harmful content?

**Level 3 — Human evaluation:**
- Domain experts rate a sample of outputs
- A/B testing in production (real user satisfaction)

**Level 4 — Regression testing:**
- Golden dataset of (input, expected_output) pairs
- Run on every prompt/model change
- Track metrics over time (dashboard)

### What is the difference between evaluation and testing for AI systems?
**Level:** Mid

- **Evaluation:** Measures quality on a benchmark. "How good is this model at summarization?"
- **Testing:** Catches regressions and edge cases. "Does changing the prompt break existing functionality?"

Testing framework:
```python
# test_chatbot.py
def test_refuses_harmful_requests():
    response = chatbot("How do I hack into...")
    assert "I can't help" in response or "I cannot assist" in response

def test_answers_product_questions():
    response = chatbot("What is the return policy?")
    assert "30 days" in response
    assert response.sources  # must cite a source

def test_handles_empty_context():
    response = chatbot("Tell me about quantum computing")  # not in knowledge base
    assert "I don't have information" in response
```

### How do you detect and measure hallucination?
**Level:** Senior

Hallucination detection methods:

1. **Faithfulness scoring:** Check if each claim in the output is supported by the context (NLI models or LLM-as-judge)
2. **Self-consistency:** Generate multiple responses. If they contradict each other, likely hallucinating.
3. **Source attribution:** Require the model to cite sources. Verify citations exist and support the claim.
4. **Factual grounding:** For known-fact domains, compare against a ground truth database.

Metrics:
- **Faithfulness rate:** % of claims supported by provided context
- **Hallucination rate:** % of responses containing unsupported claims
- **Attribution accuracy:** % of cited sources that actually support the claim

---

## Production & Infrastructure

### How do you serve LLMs in production?
**Level:** Senior

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **API providers** (OpenAI, Anthropic) | Starting out, variable load | No infra, latest models | Cost at scale, latency, vendor lock-in |
| **vLLM** | Self-hosted, high throughput | PagedAttention, continuous batching, fast | Requires GPUs, operational overhead |
| **TGI** (HuggingFace) | Self-hosted, easy setup | Good defaults, HF integration | Less flexible than vLLM |
| **TensorRT-LLM** (NVIDIA) | Maximum performance | Optimized for NVIDIA GPUs | Complex setup, NVIDIA-only |
| **Ollama** | Local development, edge | Simple, runs on laptops | Not for production scale |

**Key optimizations for self-hosted:**
- **Quantization:** GPTQ, AWQ, GGUF — reduce model size 2-4× with minimal quality loss
- **Continuous batching:** Process multiple requests simultaneously
- **KV cache optimization:** PagedAttention (vLLM) reduces memory waste
- **Speculative decoding:** Use small model to draft, large model to verify (2-3× speedup)

### How do you handle LLM latency in production?
**Level:** Senior

| Technique | Latency Reduction | Complexity |
|-----------|-------------------|------------|
| **Streaming** | Perceived latency (TTFT) | Low |
| **Caching** (semantic cache) | 0ms for cache hits | Medium |
| **Smaller model** for simple tasks | 2-5× faster | Low |
| **Prompt optimization** | Reduce input tokens = faster | Low |
| **Speculative decoding** | 2-3× end-to-end | High |
| **Model distillation** | Train small model to mimic large | High |

**Semantic caching:** Cache by meaning, not exact match. Embed the query, check if a similar query was answered recently.

```python
# Semantic cache pseudocode
cached = cache.search(embed(query), threshold=0.95)
if cached:
    return cached.response  # Cache hit
else:
    response = llm.generate(query)
    cache.store(embed(query), response)
    return response
```

### How do you monitor LLM applications in production?
**Level:** Senior

Key metrics:
- **Latency:** TTFT (time to first token), total generation time, p50/p95/p99
- **Cost:** Tokens consumed per request, cost per query, daily spend
- **Quality:** LLM-as-judge scores on sample of responses, user feedback (thumbs up/down)
- **Errors:** Rate of refusals, hallucinations, format errors, timeouts
- **Usage:** Requests per minute, unique users, token distribution

Tools: LangSmith, Arize, Weights & Biases, Helicone, custom OpenTelemetry traces.

---

## Safety, Ethics & Guardrails

### How do you implement guardrails for an LLM application?
**Level:** Senior

Guardrails = constraints that prevent harmful, off-topic, or incorrect outputs.

**Input guardrails:**
- Content classification (toxic, PII, off-topic)
- Input length limits
- Rate limiting per user
- Prompt injection detection

**Output guardrails:**
- Schema validation (JSON, structured output)
- Content filtering (harmful, biased, PII leakage)
- Fact-checking against source documents (RAG faithfulness)
- Confidence thresholds (if model uncertainty is high → fallback)

**Tools:** Guardrails AI, NeMo Guardrails (NVIDIA), LLM Guard, custom classifiers.

### How do you handle PII in LLM applications?
**Level:** Senior

LLMs can memorize and leak PII from training data or user inputs. Mitigations:

1. **Input masking:** Detect and replace PII before sending to LLM (Presidio, custom NER)
2. **Output scanning:** Check LLM output for PII before returning to user
3. **Data governance:** Don't send sensitive data to third-party APIs — use self-hosted models
4. **Logging:** Scrub PII from logs and observability traces
5. **Retention policies:** Delete conversation history per compliance requirements (GDPR, HIPAA)

### What are the ethical considerations for deploying AI in production?
**Level:** Mid

- **Bias:** Models reflect training data biases. Test across demographics. Monitor for disparate impact.
- **Transparency:** Users should know they're talking to an AI. Disclose AI-generated content.
- **Accountability:** Have a human-in-the-loop for high-stakes decisions (hiring, medical, legal).
- **Data privacy:** Minimize data collection. Honor deletion requests. Comply with regulations.
- **Environmental cost:** Large models have significant carbon footprint. Right-size your models.

---

## System Design for AI

### Design a customer support chatbot with RAG.
**Level:** Senior

**Requirements:**
- Answer questions from product documentation and past support tickets
- Escalate to human when confidence is low
- Multi-turn conversation with context
- Handle 10K concurrent users

**Architecture:**
```
User → API Gateway → Chat Service → Conversation Memory (Redis)
                                         ↓
                               Query Rewriter (LLM) → Hybrid Search (Vector + BM25)
                                                              ↓
                                                    Reranker → Top 5 chunks
                                                              ↓
                                                    LLM (with system prompt + context + history)
                                                              ↓
                                                    Output Guardrails → Response
                                                         ↓ (low confidence)
                                                    Human Escalation Queue
```

**Key decisions:**
- **Conversation memory:** Last 5-10 turns in Redis, summarize older turns
- **Query rewriting:** LLM reformulates the latest message with conversation context for better retrieval
- **Confidence routing:** If the model says "I'm not sure" or retrieval score is low → escalate
- **Feedback loop:** Human agent responses become training data for improving the system

### Design a code review AI assistant.
**Level:** Staff

**Architecture:**
```
PR Webhook → Diff Parser → Chunk Code Changes
                                ↓
                    For each chunk: LLM Review (parallel)
                    - Code quality issues
                    - Security vulnerabilities
                    - Performance concerns
                    - Style violations
                                ↓
                    Aggregator → Deduplicate → Prioritize
                                ↓
                    Post comments on PR (via GitHub API)
```

**Key decisions:**
- **Context:** Include full file (not just diff) for understanding. Include related files (imports, tests).
- **Model selection:** Use expensive model (Claude/GPT-4) for complex logic, fast model for style issues.
- **Rate limiting:** Don't flood PRs with comments. Max 5-10 actionable comments per review.
- **Confidence filter:** Only post comments where model confidence is high. Low-confidence → suggest, don't assert.
- **Learning:** Track which comments developers accept/dismiss. Use as signal to improve prompts.

---

## Scenario-Based Questions

### "Your RAG system returns correct documents but the LLM still gives wrong answers. What do you do?"
**Level:** Senior

Systematic debugging:
1. **Check the prompt:** Is the instruction clear? "Only answer based on the provided context."
2. **Check chunk size:** If chunks are too large, relevant info gets buried. Try smaller chunks.
3. **Lost in the middle:** Move most relevant chunks to the beginning and end of context.
4. **Check temperature:** Lower to 0 for factual tasks.
5. **Check model:** Try a stronger model. Some models are better at following context.
6. **Add chain-of-thought:** "First, find the relevant information in the context. Then, formulate your answer."
7. **Structured extraction:** Instead of free-form answer, extract specific fields from context first.

### "You need to reduce LLM API costs by 80% without significant quality loss."
**Level:** Senior

1. **Semantic caching:** Cache frequent queries (can save 30-50% alone)
2. **Model routing:** Use cheap/fast model for simple queries, expensive model for complex ones. Classify query complexity first.
3. **Prompt optimization:** Remove unnecessary tokens. Shorter system prompts. Batch similar requests.
4. **Fine-tune a smaller model:** Distill GPT-4 quality into Llama 3 8B for your specific task. 10-100× cheaper inference.
5. **Reduce output tokens:** Set max_tokens appropriately. Use structured output to avoid verbose responses.
6. **Pre-compute:** For predictable queries (FAQ), pre-generate and cache all answers.

### "A user reports that the AI assistant revealed another user's personal information."
**Level:** Senior

**Immediate response:**
1. **Contain:** Disable the feature or add a blanket PII filter immediately
2. **Investigate:** Check logs — was PII in the training data, RAG context, or conversation history?
3. **Identify scope:** How many users affected? What data was exposed?

**Root cause analysis:**
- If in RAG context → fix document ingestion to strip PII, add access controls per user
- If in conversation history → ensure sessions are isolated, clear history between users
- If memorized by the model → this is harder, add output PII scanning

**Prevention:**
- PII detection on all inputs AND outputs
- Tenant isolation for multi-user systems
- Regular audits of the knowledge base for PII
- Compliance review (GDPR breach notification if applicable)

---

## Related Repos

| Repo | Description |
|------|-------------|
| [devops-interview-questions](https://github.com/atryx/devops-interview-questions) | 100+ DevOps interview questions with answers |
| [system-design-interview-prep](https://github.com/atryx/system-design-interview-prep) | System design frameworks and real questions |
| [kubernetes-interview-questions](https://github.com/atryx/kubernetes-interview-questions) | Deep Kubernetes interview prep |
| [cloud-architect-interview-questions](https://github.com/atryx/cloud-architect-interview-questions) | Cloud architect interview prep |
| [what-they-dont-tell-you-about-llms](https://github.com/atryx/what-they-dont-tell-you-about-llms) | LLM gotchas from production |
| [ai-engineer-roadmap-2026](https://github.com/atryx/ai-engineer-roadmap-2026) | AI engineer learning path |
| [langchain-vs-llamaindex](https://github.com/atryx/langchain-vs-llamaindex) | LangChain vs LlamaIndex comparison |

---

## Contributing

Got an AI engineering interview question we missed? PRs welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

[MIT](LICENSE)

---

**If this helped you prepare, give it a ⭐ — it helps others find it too.**

[![Sponsored by GitAds](https://gitads.dev/v1/ad-serve?source=atryx/ai-engineer-interview-questions@github)](https://gitads.dev/v1/ad-track?source=atryx/ai-engineer-interview-questions@github)
