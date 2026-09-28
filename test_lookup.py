import sqlite3
import os
from app.services.embeddings import generate_text_embedding, cosine_similarity

def find_matching_images(user_query: str):
    conn = sqlite3.connect("catalog_qc.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Query all processed images in the catalog
    cursor.execute("""
        SELECT p.file_path, t.subject, t.category, t.color, t.material, t.caption
        FROM photos p
        JOIN photo_tags t ON p.id = t.photo_id
        WHERE p.status = 'PROCESSED'
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("\n⚠ No processed images found in the database!")
        return

    print(f"\n[QUERY] Generating vector embedding for query: '{user_query}'...")
    query_embedding = generate_text_embedding(user_query)

    results = []

    # Calculate similarity score against candidate images
    for row in rows:
        photo_context = (
            f"Product: {row['color']} {row['material']} {row['subject']} ({row['category']}). "
            f"Description: {row['caption']}"
        )
        photo_embedding = generate_text_embedding(photo_context)
        score = cosine_similarity(query_embedding, photo_embedding)

        results.append({
            "file_path": row["file_path"],
            "score": score,
            "category": row["category"],
            "context": photo_context
        })

    # Sort candidates by similarity score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "="*70)
    print(f"  SEMANTIC SEARCH TOP RESULTS FOR: '{user_query}'")
    print("="*70)

    top_n = min(3, len(results))
    for i in range(top_n):
        match = results[i]
        rank_label = f"[{i+1}] WINNER (#1)" if i == 0 else f"[{i+1}] RUNNER-UP (#{i+1})"
        print(f"\n{rank_label}")
        print(f"    Similarity Score : {match['score']:.4f}")
        print(f"    File Path        : {match['file_path']}")
        print(f"    Extracted Tags   : {match['context']}")

    print("\n" + "="*70)

    # Automatically open #1 winner
    best_match_path = results[0]["file_path"]
    if os.path.exists(best_match_path):
        print(f"[ACTION] Opening top matching image (#1): {best_match_path}...")
        os.startfile(best_match_path)
    else:
        print(f"⚠ File not found at path: {best_match_path}")

    # Prompt user if they want to open option #2 or #3
    while True:
        choice = input(f"\nOpen another candidate? Enter option number (1-{top_n}) or press Enter to continue: ").strip()
        if not choice:
            break
        if choice.isdigit() and 1 <= int(choice) <= top_n:
            idx = int(choice) - 1
            selected_path = results[idx]["file_path"]
            if os.path.exists(selected_path):
                print(f"[ACTION] Opening candidate #{choice}: {selected_path}...")
                os.startfile(selected_path)
            else:
                print(f"⚠ File not found at path: {selected_path}")
        else:
            print(f"Invalid choice. Please enter a number between 1 and {top_n}.")

if __name__ == "__main__":
    print("="*70)
    print("      E-COMMERCE REVERSE IMAGE LOOKUP & SEMANTIC SEARCH TESTER")
    print("="*70)
    
    while True:
        title_input = input("\nEnter product title/description (or type 'exit' to quit): ").strip()
        if title_input.lower() == 'exit' or not title_input:
            break

        find_matching_images(title_input)