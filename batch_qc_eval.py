import glob
import os
import time
import requests

BASE_URL = "http://127.0.0.1:8000/api/v1"
IMAGE_DIR = "D:/flyrank ai/ecommerce catalog/image data"

TEST_LISTINGS = {
    "bags": "Casual outdoor backpack with zippered compartments and padded straps",
    "sneakers": "Athletic running sneakers with rubber sole and breathable mesh",
    "jackets": "Warm winter jacket coat with zipper closure and soft lining",
    "phone covers": "Protective mobile phone cover case with shock absorption",
}


def wait_for_photos_processed(photo_ids: list[str], max_wait_sec: int = 120):
    """Polls database via GET status endpoint until all queued photos are PROCESSED or FAILED."""
    print(
        "\n[STEP 2] Polling photo status until background processing completes..."
    )
    start_time = time.time()

    while time.time() - start_time < max_wait_sec:
        pending_count = 0
        failed_count = 0
        processed_count = 0

        for pid in photo_ids:
            res = requests.get(f"{BASE_URL}/photos/{pid}")
            if res.status_code == 200:
                status = res.json().get("status")
                if status == "PENDING":
                    pending_count += 1
                elif status == "FAILED":
                    failed_count += 1
                elif status == "PROCESSED":
                    processed_count += 1

        if pending_count == 0:
            print(
                f"  ✓ Processing finished! (PROCESSED: {processed_count}, FAILED: {failed_count})"
            )
            return

        print(
            f"  ... {pending_count} photos still pending ({processed_count} done). Waiting 5 seconds..."
        )
        time.sleep(5)

    print("  ⚠ Timeout reached! Proceeding with evaluation...")


def run_batch_evaluation():
    print("=" * 60)
    print("      E-COMMERCE CATALOG QC ENGINE: BATCH EVALUATION")
    print("=" * 60)

    # Gather image files
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(
        os.path.join(IMAGE_DIR, "*.png")
    )

    if not image_paths:
        print(f"No image files found in {IMAGE_DIR}. Please check the path!")
        return

    print(f"\nFound {len(image_paths)} catalog images across categories.")

    ingested_records = []
    photo_ids = []

    # 1. Ingest/Retry photos selectively (Skipping cached PROCESSED photos)
    print("\n[STEP 1] Ingesting photos (Skipping already PROCESSED items)...")
    for path in image_paths:
        clean_path = path.replace("\\", "/")
        response = requests.post(
            f"{BASE_URL}/photos/ingest", json={"file_path": clean_path}
        )

        if response.status_code == 202:
            data = response.json()
            photo_id = data["photo_id"]
            status = data.get("status")

            ingested_records.append((photo_id, clean_path))
            photo_ids.append(photo_id)

            if status == "PROCESSED":
                print(f"  ⚡ Already Cached: {os.path.basename(clean_path)}")
            else:
                print(
                    f"  ✓ Queued for Processing: {os.path.basename(clean_path)} -> ID: {photo_id[:8]}..."
                )
                # Only sleep if we actually queued a new task to Gemini to preserve API quota
                time.sleep(5)
        else:
            print(f"  ✗ Failed to register: {os.path.basename(clean_path)}")

    # 2. Poll until background workers complete
    wait_for_photos_processed(photo_ids)

    # 3. Perform Catalog QC Verification
    print("\n[STEP 3] Running Quality Control Verification Engine...")
    print("-" * 60)

    total_evals = 0
    matches = 0
    mismatches = 0

    for photo_id, path in ingested_records:
        filename = os.path.basename(path).lower()

        # Determine category from filename
        category = None
        for cat in TEST_LISTINGS.keys():
            if cat in filename:
                category = cat
                break

        if not category:
            continue

        # Test A: Test with MATCHING listing title
        correct_title = TEST_LISTINGS[category]
        res = requests.post(
            f"{BASE_URL}/qc/verify",
            json={"photo_id": photo_id, "listing_title": correct_title},
        )

        if res.status_code == 200:
            qc_res = res.json()
            verdict = qc_res.get("verdict", "UNKNOWN")
            score = qc_res.get("similarity_score", 0.0)
            total_evals += 1
            print(
                f"MATCH TEST   | {filename[:15]:<15} | Score: {score:.4f} | Result: {verdict}"
            )
            if verdict == "MATCH":
                matches += 1
        else:
            err_detail = res.json().get("detail", "Error")
            print(
                f"MATCH TEST   | {filename[:15]:<15} | ERROR: {err_detail[:45]}"
            )

        # Test B: Test with MISMATCHING listing title
        wrong_category = "jackets" if category != "jackets" else "sneakers"
        wrong_title = TEST_LISTINGS[wrong_category]

        res_mismatch = requests.post(
            f"{BASE_URL}/qc/verify",
            json={"photo_id": photo_id, "listing_title": wrong_title},
        )

        if res_mismatch.status_code == 200:
            qc_res = res_mismatch.json()
            verdict = qc_res.get("verdict", "UNKNOWN")
            score = qc_res.get("similarity_score", 0.0)
            total_evals += 1
            print(
                f"MISMATCH TEST| {filename[:15]:<15} | Score: {score:.4f} | Result: {verdict}"
            )
            if verdict == "MISMATCH":
                mismatches += 1
        else:
            err_detail = res_mismatch.json().get("detail", "Error")
            print(
                f"MISMATCH TEST| {filename[:15]:<15} | ERROR: {err_detail[:45]}"
            )

    # Summary Metrics
    print("=" * 60)
    print("                    BATCH QC SUMMARY")
    print("=" * 60)
    print(f"Total Evaluated Comparisons: {total_evals}")
    print(f"Correct MATCH Identifications: {matches}")
    print(f"Correct MISMATCH Detections:   {mismatches}")
    if total_evals > 0:
        accuracy = ((matches + mismatches) / total_evals) * 100
        print(f"Engine QC Accuracy Rate:       {accuracy:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_batch_evaluation()