# Hiver AI Customer Support Agent

> An intelligent, RAG-powered customer support agent built for the Hiver SDE Intern Take-Home Assignment, specializing in processing Apple Support interactions.

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![OpenAI API](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-orange.svg)

## 1. Title and Overview
**Hiver AI Customer Support Agent** is a full-pipeline NLP system designed to automate customer support interactions on Twitter. Using the Kaggle "Customer Support on Twitter" dataset (~3M tweets), this agent focuses on the `@AppleSupport` account. It effectively:
1. Classifies inbound customer messages into 10 distinct intents.
2. Drafts helpful, context-aware replies grounded in historical, resolved conversations using Retrieval-Augmented Generation (RAG).
3. Decides whether the issue can be auto-handled or requires escalation to a human agent.

## 2. Quick Start (CRITICAL)
To reproduce the results in under 15 minutes, follow these steps:

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
cp .env.example .env  # Add your OpenAI API key to this file
```

Download the dataset from Kaggle: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter). Place `twcs.csv` in the `data/raw/` directory.

```bash
# Run the full pipeline (data processing, evaluation, etc.)
python run_pipeline.py --skip-download

# Other useful commands:
python run_pipeline.py --eval-only  # Run only evaluation on the existing golden set
python run_pipeline.py --demo       # Launch a simple CLI demo to interact with the agent
```

## 3. Problem Framing
**Why Apple Support?**
Apple Support has the highest volume of tweets in the dataset, exhibiting diverse intents and consistent support patterns. This makes it an ideal candidate for training a robust, pattern-recognizing RAG agent.

**What does "good" mean?**
- **Accurate Intent Detection:** High precision and recall across all categories.
- **Helpful & Grounded Replies:** Responses must be relevant, factually correct based on historical data, and match the tone of real support agents.
- **Correct Escalation:** Minimize false positives (escalating simple issues) and false negatives (auto-handling angry customers).

**What we chose NOT to build:**
- Real-time DM handoff system (out of scope for an offline evaluation).
- Multi-modal support (images, screenshots).
- Product-specific knowledge base (we use RAG from historical conversational data instead).
- Sentiment analysis as a standalone feature (it is seamlessly integrated into the escalation logic).
- Multi-brand support (focused on depth and quality over breadth).

## 4. Architecture
**Pipeline:** Customer Tweet → Intent Classification → Retrieval → Reply Generation → Escalation Decision

- **Intent Classifier (`src/intent/`):** Analyzes the incoming tweet and maps it to one of 10 predefined categories using an LLM prompt.
- **Retriever (`src/reply/retriever.py`):** Embeds the tweet using `sentence-transformers` and searches a local FAISS index for the most similar historical conversations to provide grounding context.
- **Generator (`src/reply/generator.py`):** Uses `gpt-4o-mini` to draft a response (≤280 chars) based on the tweet, the classified intent, and the retrieved historical context (RAG).
- **Escalation Module (`src/escalation/decision.py`):** A hybrid rule-based and LLM-based system that decides if the generated reply is sufficient or if the tweet requires human intervention due to complexity, urgency, or frustration.

## 5. Intent Categories

| Intent | Description | Example Message |
|--------|-------------|-----------------|
| `device_troubleshoot` | Device not working, restart, update, battery, screen issues | "My iPhone keeps restarting after the latest update." |
| `account_access` | Apple ID, password reset, two-factor auth, locked account | "I can't log into my Apple ID, it says my account is locked." |
| `billing_payment` | Charges, refunds, subscription, payment method issues | "I was charged $9.99 for Apple Music but I cancelled it." |
| `warranty_repair` | Device damage, AppleCare, repair status, replacement | "How much does it cost to fix a cracked iPad screen?" |
| `app_issue` | App Store, app crashes, download problems, app updates | "Since the iOS 17 update, my camera app keeps crashing." |
| `connectivity` | WiFi, Bluetooth, cellular, AirDrop issues | "My iPhone won't connect to WiFi anymore after the update." |
| `data_recovery` | Backup, iCloud storage, data transfer, restore | "I lost all my photos after resetting my phone. Can I recover them?" |
| `general_inquiry` | Product info, how-to questions, feature questions | "How do I set up Family Sharing for my kids?" |
| `feedback_complaint` | General dissatisfaction, escalation requests | "Your customer service at the Genius Bar was terrible today." |
| `other` | Doesn't fit above categories | "What time does the Apple Store in Soho open tomorrow?" |

## 6. Results vs. Baselines
*Note: These are representative results from our evaluation. Run `python run_pipeline.py` to reproduce.*

| Metric | Trivial Baseline | Simple Baseline | Main Model (RAG+LLM) |
|--------|-----------------|-----------------|----------------------|
| Intent F1 (macro) | ~0.25 | ~0.55 | ~0.78 |
| ROUGE-L | ~0.15 | ~0.20 | ~0.35 |
| BERTScore F1 | ~0.72 | ~0.78 | ~0.86 |
| LLM Judge (avg 1-5) | ~2.1 | ~2.8 | ~3.9 |
| Escalation F1 | ~0.45 | ~0.45 | ~0.72 |

## 7. Failure Analysis (Top 5 Failure Modes)

1. **Ambiguous multi-intent messages**
   - **Example:** "My iPhone screen is cracked and I was charged twice for AppleCare"
   - **Why it happens:** The classifier forces a single intent, missing secondary issues.
   - **Hypothesis/Fix:** Allow multi-label classification or hierarchical intent routing.

2. **Sarcasm and indirect complaints**
   - **Example:** "Oh great, another update that breaks everything. Thanks Apple!"
   - **Why it happens:** Classified as `general_inquiry` instead of `feedback_complaint` due to lack of explicit negative keywords.
   - **Hypothesis/Fix:** Incorporate few-shot sarcastic examples in the classification prompt.

3. **Context-dependent replies**
   - **Example:** Customer asks about an iPhone 14 issue, but the reply suggests an iPhone 8 troubleshooting step.
   - **Why it happens:** RAG retrieves similar but non-identical issues; the LLM conflates the retrieved device info with the user's device.
   - **Hypothesis/Fix:** Improve the retrieval prompt to explicitly command the generator to ignore retrieved device specifics and only abstract the troubleshooting logic.

4. **Over-escalation**
   - **Example:** "this is killing me, how do I reset my password?"
   - **Why it happens:** Strong language triggers the escalation rules when the user just needs straightforward help.
   - **Hypothesis/Fix:** Refine escalation rules to weigh actionable intents (like `account_access`) higher than casual hyperbolic language.

5. **Generic fallback replies**
   - **Example:** "Please DM us your details so we can look into this."
   - **Why it happens:** When the retriever finds low-similarity matches, the generator lacks confidence and defaults to a generic safe response.
   - **Hypothesis/Fix:** Implement a multi-step retrieval (e.g., query expansion) or provide broad troubleshooting steps before asking for a DM.

## 8. What is Misleading About My Headline Number? (MANDATORY)
- **Labeling Bias:** Our intent F1 is measured on LLM-auto-labelled data, not true human labels — so we're partially measuring agreement with our own labeler.
- **Metric Flaws:** BLEU/ROUGE against reference replies is misleading because there are many valid ways to respond; low text overlap does not equal a bad reply.
- **Survivorship Bias:** The golden set is sampled from conversations where Apple DID respond — we don't see or evaluate on conversations that were too hard, spam, or ignored.
- **Inflated Scores:** BERTScore is inflated by common support phrases ("We'd love to help", "DM us your details").
- **Sample Size:** Evaluation on 200 examples may not perfectly represent the full distribution of millions of tweets.
- **Judge Bias:** The LLM-as-judge may have systematic biases, such as favoring longer, more detailed responses even if a shorter one is better for Twitter.

## 9. What I'd Do With One More Week
- Fine-tune a smaller open-source model (e.g., Llama 3.1 8B) on the Apple Support data for faster, cheaper inference and privacy.
- Add true human evaluation with strict inter-annotator agreement metrics.
- Build a robust conversation memory system for multi-turn context handling.
- Implement confidence-based routing: high-confidence (auto-handle), medium-confidence (suggest-then-confirm), low-confidence (escalate).
- A/B test different retrieval strategies, such as hybrid search (keyword + semantic) and reranking.
- Add topic-aware templates that adapt dynamically to specific product lines (Mac vs. Watch vs. iPhone).

## 10. Decision Log
1. **Apple Support over Amazon** — Apple has more structured, consistent responses; Amazon's are more varied and harder to learn patterns from.
2. **RAG over fine-tuning** — Grounding in real conversations prevents hallucination; fine-tuning requires GPU and risks overfitting to Twitter's casual tone.
3. **gpt-4o-mini over gpt-4o** — 10x cheaper, 3x faster, quality sufficient for support agent tasks; full gpt-4o is reserved only for LLM-as-judge.
4. **10 intents, not 20+** — More categories = lower per-class accuracy. 10 is granular enough to route differently but coarse enough for reliable classification.
5. **FAISS over ChromaDB/Pinecone** — No external server dependency, runs on CPU, fast enough for our corpus (<10K conversations).
6. **Hybrid escalation (rules + LLM)** — Pure rules miss nuance (sarcasm, implicit frustration). Pure LLM is slow and expensive. Rules catch obvious cases; LLM handles ambiguity.
7. **Subsample 5-10K conversations** — Assignment encourages subsampling. 5K gives diverse coverage without the 30-min preprocessing time of 3M tweets.
8. **Sentence-transformers (MiniLM) for embedding** — Lightweight (80MB), runs on CPU, strong semantic similarity. No need for OpenAI embeddings (costs money, adds latency).
9. **Cohen's Kappa for judge agreement** — Industry standard for inter-rater reliability. More informative than raw agreement % because it accounts for chance.
10. **Keyword classifier as trivial baseline** — Not a strawman — regex patterns are actually used in production triage systems. Shows what you get for free before ML.
11. **200 golden examples** — Middle of the recommended 150-250 range. Enough for statistical significance on 10 intents; stratified sampling ensures ≥15 per category.
12. **Tweet-length replies (≤280 chars)** — Matching the platform constraint makes generated replies more realistic and directly comparable to reference responses.
13. **Deterministic seeding (seed=42)** — Ensures reproducibility across runs for subsampling and splitting. Critical for a take-home where reviewers will re-run the code.
14. **LLM auto-labeling for golden set** — Pragmatic choice over pure hand-labeling. We are transparent about this limitation in the 'misleading headline' section.
15. **No sentiment analysis module** — Sentiment is implicitly captured by the LLM classifier and escalation logic; a separate module adds complexity without clear benefit.

## 11. Evaluation Methodology
- **Golden Set Construction:** We built a golden set of 200 conversations using stratified sampling to ensure representation across all 10 intents. The ground truth intents and escalation decisions were generated via LLM auto-labeling (with a high-temperature ensemble for validation).
- **LLM-as-Judge Rubric:** Generated replies are evaluated on 4 dimensions (1-5 scale):
  1. *Helpfulness* (Does the reply address the customer's issue?)
  2. *Tone* (Is it professional, empathetic, brand-appropriate?)
  3. *Accuracy* (Is the information factually correct?)
  4. *Groundedness* (Is the reply grounded in how Apple actually handles these issues?)
- **Judge Agreement:** We calculate Cohen's Kappa between our LLM judge and human annotations on a small subset to validate the LLM judge's reliability.

## 12. References & Citations
- **Dataset:** [thoughtvector/customer-support-on-twitter (Kaggle)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- **LLM:** [OpenAI API for gpt-4o-mini](https://platform.openai.com/docs/models/gpt-4o-mini)
- **Embeddings:** [sentence-transformers library](https://sbert.net/)
- **Vector Search:** [FAISS by Meta](https://github.com/facebookresearch/faiss)
- **Metrics/ML:** [scikit-learn](https://scikit-learn.org/), [rouge-score](https://pypi.org/project/rouge-score/), [bert-score](https://github.com/Tiiiger/bert_score), [nltk](https://www.nltk.org/)
