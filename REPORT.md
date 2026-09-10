# Hiver SDE Intern — AI Support Agent

## 1. Problem Framing

### Selected Brand
**SpotifyCares** — Spotify's customer support account on Twitter.

### System Behavior
The agent receives a customer support tweet and must:
1. Classify the message into a data-derived intent category
2. Retrieve similar historical support cases
3. Draft a grounded reply using evidence from historical resolutions
4. Decide whether to auto-handle or escalate to a human agent
5. Provide an explicit reason for the escalation decision

### What "Good" Means
- Correctly identifies the customer's intent
- Retrieves relevant historical cases
- Generates grounded, helpful replies
- Escalates appropriately (especially for safety-critical issues)
- Provides transparent, auditable decisions

### What Was Deliberately NOT Built
- A full production deployment system
- Real-time Twitter integration
- Multi-turn conversation handling (simplified to single-turn)
- Fine-tuned models (keeping it explainable)
- Complex ML pipelines (preferring simplicity)

---

## 2. Dataset and Sampling

### Dataset
Kaggle's `thoughtvector/customer-support-on-twitter` dataset (2.8M rows).

### Brand Extraction
Filtered for `author_id == "SpotifyCares"` with `inbound == False` (outgoing replies).

**Results:**
- Total rows: 2,811,774
- SpotifyCares outgoing replies: 43,265
- Reconstructed conversations: 43,092
- Flattened cases: 57,822
- Broken relationships: 0.4%

### Conversation Reconstruction
Used `tweet_id`, `response_tweet_id`, and `in_response_to_tweet_id` to reconstruct conversation threads. Built parent-child relationships following the reply chain upward.

### Development/Golden Separation
- **Conversation-level split** (not tweet-level)
- 85% development pool (49,298 cases)
- 15% golden candidate pool (8,524 candidates)
- Deterministic random seed (42)
- **No leakage**: conversations appear in only one partition

### Golden Set Sampling
- Stratified by intent category (25 examples per intent)
- 200 total candidates
- Mixed difficulty levels (EASY, MEDIUM, HARD)

### Manual Labelling
**STATUS: PENDING HUMAN LABELS**

The golden set uses heuristic labels for pipeline testing. These are NOT human labels. The labeling tool (`scripts/label_golden.py`) is ready for manual annotation.

---

## 3. System Architecture

```
Incoming Customer Message
        ↓
┌─────────────────────┐
│  Intent Classifier  │ ← LLM + Rule-based hints
│  (8 categories)     │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ Historical Case     │ ← Sentence embeddings
│ Retrieval (top-5)   │   (all-MiniLM-L6-v2)
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ Grounded Reply      │ ← LLM with retrieved evidence
│ Generator           │   + safety constraints
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ Escalation Policy   │ ← Hybrid rules
│ (explicit rules)    │   + keyword detection
└─────────┬───────────┘
          ↓
    AUTO_HANDLE / ESCALATE
```

### Key Components

1. **Intent Classification** (`src/intents.py`)
   - LLM-based with rule-based augmentation
   - 8 data-derived categories
   - Confidence scoring

2. **Retrieval** (`src/retrieval.py`)
   - Sentence-transformers embeddings
   - Cosine similarity search
   - Local caching

3. **Reply Generation** (`src/agent.py`)
   - Grounded in retrieved evidence
   - Safety constraints (no invented policies, no PII exposure)
   - Provenance tracking

4. **Escalation Policy** (`src/escalation.py`)
   - Explicit keyword-based triggers
   - Confidence threshold checks
   - Retrieval evidence validation

---

## 4. Evaluation

### Metrics

**Intent Classification:**
- Accuracy
- Macro F1 (headline metric — handles class imbalance)
- Weighted F1
- Per-class precision/recall/F1
- Confusion matrix

**Escalation:**
- Precision, Recall, F1 (ESCALATE as positive class)
- Unsafe auto-handle rate (critical safety metric)
- Auto-handle rate

**Reply Quality:**
- LLM-as-judge rubric (1-5 scale):
  - Correctness
  - Relevance
  - Groundedness
  - Helpfulness
  - Brand consistency
  - Safety

### Baselines

1. **Trivial Baseline**: Always predicts majority intent, always escalates
2. **TF-IDF Baseline**: TF-IDF + LogisticRegression for intent, nearest-neighbor for reply

### LLM Judge
- Uses rubric-based evaluation
- Deterministic (temperature=0)
- Caches results for reproducibility
- Validates against human scores (PENDING)

---

## 5. Results

### TF-IDF Baseline Results (Heuristic Labels)

| Metric | Value |
|--------|-------|
| Intent Accuracy | 0.77 |
| Intent Macro F1 | 0.78 |
| Intent Weighted F1 | 0.78 |
| Escalation Precision | 0.39 |
| Escalation Recall | 0.48 |
| Escalation F1 | 0.43 |
| Unsafe Auto-handle Rate | 0.52 |

**Note:** These results use heuristic labels, NOT human labels. Actual results pending human annotation.

---

## 6. Failure Analysis

Based on the TF-IDF baseline evaluation:

1. **Unsafe Auto-handling (52%)**: The baseline auto-handles messages that should be escalated
2. **Intent Confusion**: Similar intents (e.g., BILLING_PAYMENT vs SUBSCRIPTION_PLAN) are confused
3. **Weak Retrieval**: TF-IDF may not capture semantic similarity well
4. **Over-escalation**: Conservative thresholds lead to unnecessary escalations
5. **Context Dependency**: Short messages without context are hard to classify

---

## 7. What is misleading about my headline number?

The headline number (0.78 Macro F1) is misleading because:

1. **Heuristic labels**: The golden set uses algorithmic labels, not human annotations. Human labels would likely shift the true accuracy.

2. **Class balance**: 25 examples per intent doesn't match natural production frequency. BILLING_PAYMENT likely occurs more often than FEATURE_REQUEST.

3. **Twitter-specific**: Twitter messages are very short, with lots of slang and abbreviations. This doesn't reflect typical support channels.

4. **Historical data**: The training data is from 2017. Spotify's policies and products have changed significantly.

5. **Single-turn**: The evaluation treats each message independently, but real support involves multi-turn conversations.

6. **No adversarial testing**: The golden set doesn't include deliberately adversarial or edge-case inputs.

7. **TF-IDF limitations**: The baseline uses simple TF-IDF, which misses semantic nuance.

---

## 8. What I would do with one more week

1. **Human annotation**: Complete manual labeling of all 200 golden examples
2. **Inter-annotator agreement**: Add a second annotator and measure Cohen's kappa
3. **LLM-based agent**: Implement the full LLM-based agent and compare to TF-IDF baseline
4. **Threshold tuning**: Use development set to optimize escalation thresholds
5. **Error analysis**: Deep-dive into specific failure cases
6. **Adversarial evaluation**: Create deliberately hard test cases
7. **Confidence calibration**: Improve confidence score reliability
8. **Multi-turn support**: Extend to handle conversation context
9. **Bootstrap confidence intervals**: Add uncertainty estimates to metrics
10. **Production readiness**: Add monitoring, logging, and alerting

---

## Appendix: Project Structure

```
hiver-support-agent/
├── README.md
├── REPORT.md
├── decision_log.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   ├── intents.yaml
│   └── settings.yaml
├── data/
│   ├── raw/twcs.csv
│   ├── processed/
│   └── golden/
├── src/
│   ├── agent.py
│   ├── conversations.py
│   ├── escalation.py
│   ├── intents.py
│   ├── llm.py
│   ├── retrieval.py
│   └── schemas.py
├── baselines/
│   ├── trivial.py
│   └── tfidf.py
├── eval/
│   ├── run_all.py
│   ├── run_quick.py
│   ├── evaluate_intents.py
│   ├── evaluate_escalation.py
│   └── judge.py
├── scripts/
│   ├── prepare_data.py
│   ├── create_golden_candidates.py
│   └── label_golden.py
└── tests/
    ├── test_conversations.py
    ├── test_preprocess.py
    ├── test_retrieval.py
    ├── test_escalation.py
    └── test_schemas.py
```
