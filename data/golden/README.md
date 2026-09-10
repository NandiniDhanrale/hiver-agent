# Golden Evaluation Set

## Overview

This directory contains the golden evaluation set for the Hiver Support Agent.

## Files

- `golden_candidates.csv` - Auto-generated candidates for labeling
- `golden_set.csv` - Human-labeled golden examples (created by labeling tool)
- `human_reply_scores.csv` - Human scores for judge validation

## Labeling Process

1. `golden_candidates.csv` is generated with stratified sampling
2. The labeling tool (`scripts/label_golden.py`) presents each example
3. A human annotator assigns:
   - `gold_intent` - The correct intent category
   - `gold_escalate` - Whether this should be escalated
   - `gold_escalation_reason` - Why escalation is needed (if applicable)
   - `reference_resolution` - Ground truth reply (if available)
   - `difficulty` - EASY / MEDIUM / HARD
   - `label_notes` - Any notes about the labeling decision

## Schema

| Column | Type | Description |
|--------|------|-------------|
| example_id | string | Unique identifier |
| conversation_id | string | Source conversation |
| customer_text | string | Customer message |
| conversation_context | string | Prior conversation turns |
| gold_intent | string | Human-labeled intent |
| gold_escalate | bool | Human-labeled escalation |
| gold_escalation_reason | string | Escalation reason |
| reference_resolution | string | Historical reply |
| difficulty | string | EASY/MEDIUM/HARD |
| label_notes | string | Annotation notes |
| human_verified | bool | True if human-reviewed |

## Target Size

~200 examples, stratified across intents and difficulty levels.
