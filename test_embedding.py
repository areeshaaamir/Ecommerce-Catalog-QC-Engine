from app.services.embeddings import generate_text_embedding, cosine_similarity

# High semantic overlap (Bags / Rucksacks)
text1 = "A black canvas backpack with zippered pockets"
text2 = "Black school rucksack for daily carry"

# Zero semantic overlap (Shoes)
text3 = "Red leather high-heel pumps"

print("Generating vector embeddings...")
emb1 = generate_text_embedding(text1)
emb2 = generate_text_embedding(text2)
emb3 = generate_text_embedding(text3)

sim_high = cosine_similarity(emb1, emb2)
sim_low = cosine_similarity(emb1, emb3)

print("\n--- RESULTS ---")
print(f"Text 1: '{text1}'")
print(f"Text 2: '{text2}'")
print(f"Cosine Similarity: {sim_high:.4f} (Expected > 0.70)\n")

print(f"Text 1: '{text1}'")
print(f"Text 3: '{text3}'")
print(f"Cosine Similarity: {sim_low:.4f} (Expected < 0.50)")