from app.services.embeddings import generate_text_embedding, cosine_similarity

# High semantic overlap
text1 = "A black canvas backpack with zippered pockets"
text2 = "Black school rucksack for daily carry"

# Low semantic overlap
text3 = "Red leather high-heel pumps"

print("Generating vector embeddings via OpenRouter...")
emb1 = generate_text_embedding(text1)
emb2 = generate_text_embedding(text2)
emb3 = generate_text_embedding(text3)

sim_high = cosine_similarity(emb1, emb2)
sim_low = cosine_similarity(emb1, emb3)

print("\n--- OPENROUTER EMBEDDING RESULTS ---")
print(f"Text 1 vs Text 2 (Bags):     Cosine Similarity = {sim_high:.4f} (Expected > 0.55)")
print(f"Text 1 vs Text 3 (High Heels): Cosine Similarity = {sim_low:.4f} (Expected < 0.40)")