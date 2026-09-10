# Decision Log

## 1. Why SpotifyCares?

**Decision:** Use SpotifyCares as the default brand.

**Why:** SpotifyCares has 43,265 outgoing replies and 43,092 reconstructed conversations. This provides sufficient data for retrieval and evaluation. Spotify is a well-known brand with diverse support issues.

**Alternative considered:** AmazonHelp (42,944 replies), AppleSupport (15,694 replies)

**Tradeoff:** AmazonHelp has slightly more data, but Spotify support issues are more varied (billing, playback, account, family plans), making for a more interesting classification task.

---

## 2. Why this intent taxonomy?

**Decision:** 8 intents: LOGIN_ACCOUNT_ACCESS, BILLING_PAYMENT, SUBSCRIPTION_PLAN, PLAYBACK_APP_ISSUE, FAMILY_STUDENT_PLAN, ACCOUNT_SECURITY, FEATURE_REQUEST, OTHER.

**Why:** Derived from inspecting actual customer messages. Each category represents a distinct support issue with different resolution patterns.

**Alternative considered:** Fewer (5) or more (12) categories

**Tradeoff:** 8 categories balance granularity (enough to be useful) with manageability (enough examples per class for evaluation).

---

## 3. Why 6-9 intents?

**Decision:** Target 8 intents.

**Why:** Too few intents (e.g., 3) would group dissimilar issues together. Too many (e.g., 15) would create classes with very few examples, making evaluation unreliable.

**Alternative considered:** 5 intents (merging similar categories)

**Tradeoff:** Splitting FAMILY_STUDENT_PLAN from SUBSCRIPTION_PLAN increases accuracy for that common support issue.

---

## 4. Why conversation-level splitting?

**Decision:** Split conversations, not individual tweets.

**Why:** Multiple tweets from the same conversation are highly correlated. Splitting at tweet level would leak information from the same conversation between train and test.

**Alternative considered:** Random tweet-level split

**Tradeoff:** Conversation-level split reduces available data slightly but prevents data leakage.

---

## 5. Why macro F1 is headline?

**Decision:** Use Macro F1 as the headline intent metric.

**Why:** The intent classes are imbalanced (BILLING_PAYMENT likely occurs more than ACCOUNT_SECURITY). Macro F1 weights all classes equally, preventing the majority class from dominating.

**Alternative considered:** Accuracy, Weighted F1

**Tradeoff:** Macro F1 is more sensitive to poor performance on rare classes, which is desirable for a support system that must handle all issue types.

---

## 6. Why embedding retrieval?

**Decision:** Use sentence-transformers embeddings for retrieval.

**Why:** Captures semantic similarity beyond keyword matching. A customer saying "I can't get in" would match a historical case about "login problems."

**Alternative considered:** TF-IDF retrieval, BM25

**Tradeoff:** Embeddings are slower but capture meaning better. TF-IDF is a good baseline but misses synonyms.

---

## 7. Why top-K value of 5?

**Decision:** Retrieve top 5 similar cases.

**Why:** Provides enough evidence for the LLM to generate a grounded reply without overwhelming it with irrelevant cases.

**Alternative considered:** Top-3 (too little evidence), Top-10 (too much noise)

**Tradeoff:** 5 cases balances evidence quality with context window limits.

---

## 8. Why explicit escalation rules?

**Decision:** Use explicit keyword-based rules, not just LLM judgment.

**Why:** LLM self-confidence is not calibrated probability. Explicit rules ensure critical issues (account compromise, payment disputes) always escalate.

**Alternative considered:** Pure LLM-based escalation

**Tradeoff:** Rules are less flexible but more predictable and auditable.

---

## 9. Why escalation recall matters?

**Decision:** Optimize for high recall on ESCALATE class.

**Why:** In customer support, missing a critical issue (unsafe auto-handle) is worse than unnecessary escalation. False negatives are more costly than false positives.

**Alternative considered:** Optimize for precision

**Tradeoff:** Higher recall means more unnecessary escalations, but fewer missed critical issues.

---

## 10. Why unsafe auto-handle rate is important?

**Decision:** Track unsafe_auto_handle_rate as a key metric.

**Why:** This measures the most dangerous failure: the system confidently handling a message that should have gone to a human. It's the primary safety metric.

**Alternative considered:** Just using overall accuracy

**Tradeoff:** Overall accuracy doesn't capture the severity of different error types.

---

## 11. Why rubric-based reply evaluation?

**Decision:** Use 6-dimension rubric for reply quality.

**Why:** Simple metrics (BLEU, ROUGE) don't capture whether a reply is actually helpful, grounded, or safe. A rubric provides structured, interpretable evaluation.

**Alternative considered:** Single overall score

**Tradeoff:** More complex to evaluate but provides actionable feedback on specific dimensions.

---

## 12. Why BLEU/ROUGE were not primary?

**Decision:** Do not use BLEU/ROUGE as primary reply metrics.

**Why:** These measure n-gram overlap, not whether the reply is correct, helpful, or safe. A reply could have low BLEU but be perfect.

**Alternative considered:** Using BLEU as a secondary metric

**Tradeoff:** BLEU is cheap to compute but not meaningful for support quality.

---

## 13. Why stratified golden-set sampling?

**Decision:** Stratify golden set by intent category.

**Why:** Ensures enough examples of each intent for per-class evaluation. Random sampling might miss rare intents entirely.

**Alternative considered:** Pure random sampling

**Tradeoff:** Stratification makes evaluation more reliable but doesn't match natural distribution.

---

## 14. Why LLM judge validation is necessary?

**Decision:** Require human-vs-LLM judge agreement.

**Why:** LLM judges can be miscalibrated. Without human validation, we can't trust LLM scores. The assignment explicitly requires this validation.

**Alternative considered:** Trust LLM judge without validation

**Tradeoff:** Human validation is expensive but necessary for credibility.

---

## 15. Why keep the project simple?

**Decision:** Simple architecture, no microservices, no databases.

**Why:** This is an internship take-home. The goal is demonstrate evaluation discipline, not build a production system. Simpler code is easier to explain and modify.

**Alternative considered:** More complex architecture with FastAPI, Redis, etc.

**Tradeoff:** Simplicity makes the project more reproducible and easier to evaluate.
