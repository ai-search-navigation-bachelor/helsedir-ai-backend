#!/usr/bin/env python3
"""Quick test to compare base model vs fine-tuned model."""
from sentence_transformers import SentenceTransformer, util

# Test queries and expected matches
test_cases = [
    {
        "query": "diabetes behandling",
        "positive": "Nasjonal faglig retningslinje for diabetes. Dette er en retningslinje. Behandling av type 2 diabetes hos voksne.",
        "negative": "Forskrift om hundehold. Dette er en veileder. Regler for hundeeiere i kommunen.",
    },
    {
        "query": "depresjon hos eldre",
        "positive": "Depresjon hos eldre pasienter. Dette er en retningslinje. Diagnostikk og behandling av depresjon i geriatrien.",
        "negative": "Kosthold for spedbarn. Dette er en veileder. Anbefalinger for ernæring det første leveåret.",
    },
]

for model_name, label in [
    ("intfloat/multilingual-e5-base", "BASE MODEL"),
    ("models/finetuned-e5-gpl", "FINE-TUNED MODEL"),
]:
    print(f"\n{'='*50}")
    print(f"  {label}: {model_name}")
    print(f"{'='*50}")

    model = SentenceTransformer(model_name)

    for tc in test_cases:
        q_emb = model.encode(f"query: {tc['query']}")
        p_emb = model.encode(f"passage: {tc['positive']}")
        n_emb = model.encode(f"passage: {tc['negative']}")

        sim_pos = float(util.cos_sim(q_emb, p_emb)[0][0])
        sim_neg = float(util.cos_sim(q_emb, n_emb)[0][0])

        print(f"\n  Query: {tc['query']}")
        print(f"  Sim positive: {sim_pos:.4f}")
        print(f"  Sim negative: {sim_neg:.4f}")
        print(f"  Margin:       {sim_pos - sim_neg:.4f}")
