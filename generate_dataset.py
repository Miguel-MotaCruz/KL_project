
from pathlib import Path
import csv
import argparse
import random
import math
from pathlib import Path
from typing import List, Tuple


def gather_triples(wikidata_dir: Path):
	"""Return a list of (subject, relation, object) triples.

	Relation values will be the verb phrases 'has gender' and 'has occupation'.
	Duplicates are removed while preserving insertion order.
	"""
	triples = []
	seen = set()

	for csv_path in sorted(wikidata_dir.glob("query_results_*.csv")):
		try:
			with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
				reader = csv.DictReader(fh)
				for row in reader:
					name = (row.get("humanLabel") or row.get("name") or "").strip()
					gender = (row.get("genderLabel") or row.get("gender") or "").strip()
					occupation = (row.get("label") or row.get("occupation") or "").strip()

					if not name:
						continue

					if gender:
						tr = (name, "has gender", gender)
						if tr not in seen:
							seen.add(tr)
							triples.append(tr)

					if occupation:
						tr = (name, "has occupation", occupation)
						if tr not in seen:
							seen.add(tr)
							triples.append(tr)

		except Exception as e:
			print(f"Warning: failed to read {csv_path}: {e}")

	return triples


def gather_triples_limited(wikidata_dir: Path, row_limit: int = 200, seed: int | None = None) -> List[Tuple[str, str, str]]:
    """
    For each CSV in wikidata_dir, sample up to `row_limit` rows that preserve the gender
    distribution (male/female) seen in that CSV. Only rows with name, gender ('male'|'female')
    and occupation are considered eligible (this follows your original acceptance logic).
    Returns a list of unique triples (name, predicate, value) aggregated across files.
    """
    if seed is not None:
        random.seed(seed)

    triples = []
    seen = set()

    for csv_path in sorted(wikidata_dir.glob("query_results_*.csv")):
        try:
            eligible = []  # list of dicts with keys name, gender, occupation
            with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    name = (row.get("humanLabel") or row.get("name") or "").strip()
                    gender = (row.get("genderLabel") or row.get("gender") or "").strip().lower()
                    occupation = (row.get("label") or row.get("occupation") or "").strip()

                    if not name:
                        continue
                    if gender not in ("male", "female"):
                        continue
                    if not occupation:
                        continue

                    eligible.append({"name": name, "gender": gender, "occupation": occupation})

            if not eligible:
                continue 

            counts = {"male": 0, "female": 0}
            for r in eligible:
                counts[r["gender"]] += 1

            total_eligible = counts["male"] + counts["female"]
            target_total = min(row_limit, total_eligible)

            if counts["male"] == 0:
                targets = {"male": 0, "female": target_total}
            elif counts["female"] == 0:
                targets = {"male": target_total, "female": 0}
            else:
                raw_targets = {
                    g: (counts[g] / total_eligible) * target_total
                    for g in ("male", "female")
                }
                floored = {g: math.floor(raw_targets[g]) for g in raw_targets}
                allocated = sum(floored.values())
                remainder = target_total - allocated

                fractions = {g: raw_targets[g] - floored[g] for g in raw_targets}
                for g, _ in sorted(fractions.items(), key=lambda kv: kv[1], reverse=True):
                    if remainder <= 0:
                        break
                    floored[g] += 1
                    remainder -= 1

                targets = floored  

            buckets = {"male": [], "female": []}
            for r in eligible:
                buckets[r["gender"]].append(r)

            random.shuffle(buckets["male"])
            random.shuffle(buckets["female"])

            selected_rows = []
            for g in ("male", "female"):
                take = min(targets[g], len(buckets[g]))
                selected_rows.extend(buckets[g][:take])

            if len(selected_rows) < target_total:
                remaining_needed = target_total - len(selected_rows)
                leftovers = []
                for g in ("male", "female"):
                    bucket = buckets[g]
                    already = set((r["name"], r["gender"], r["occupation"]) for r in selected_rows)
                    leftovers.extend([r for r in bucket if (r["name"], r["gender"], r["occupation"]) not in already])
                random.shuffle(leftovers)
                selected_rows.extend(leftovers[:remaining_needed])

            for r in selected_rows:
                name = r["name"]
                gender = r["gender"]
                occupation = r["occupation"]

                if gender:
                    tr = (name, "has gender", gender)
                    if tr not in seen:
                        seen.add(tr)
                        triples.append(tr)

                if occupation:
                    tr = (name, "has occupation", occupation)
                    if tr not in seen:
                        seen.add(tr)
                        triples.append(tr)

        except Exception as e:
            print(f"Warning: failed to read {csv_path}: {e}")

    return triples


def gather_triples_balanced(
    wikidata_dir: Path,
    per_gender: int = 100,
    seed: int | None = None,
) -> List[Tuple[str, str, str]]:
    """
    For each CSV in wikidata_dir, sample up to `per_gender` rows for each gender (male/female),
    producing balanced samples per file. If a specific CSV has fewer than `per_gender` rows
    for either gender, the per-gender target for that CSV becomes the smallest available
    gender count (so both genders are sampled equally, up to that smaller count).

    Only rows with name, gender ('male'|'female') and occupation are considered eligible.
    Returns a list of unique triples (name, predicate, value) aggregated across files.

    Note: If one gender is absent in a CSV this results in per-file per-gender target = 0
    (so no rows are sampled from that CSV) — this is the strict balanced behavior.
    """
    if seed is not None:
        random.seed(seed)

    triples: List[Tuple[str, str, str]] = []
    seen = set()

    for csv_path in sorted(wikidata_dir.glob("query_results_*.csv")):
        try:
            eligible = []  
            with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    name = (row.get("humanLabel") or row.get("name") or "").strip()
                    gender = (row.get("genderLabel") or row.get("gender") or "").strip().lower()
                    occupation = (row.get("label") or row.get("occupation") or "").strip()

                    if not name:
                        continue
                    if gender not in ("male", "female"):
                        continue
                    if not occupation:
                        continue

                    eligible.append({"name": name, "gender": gender, "occupation": occupation})

            if not eligible:
                continue

            # Bucket rows by gender
            buckets = {"male": [], "female": []}
            for r in eligible:
                buckets[r["gender"]].append(r)

            counts = {"male": len(buckets["male"]), "female": len(buckets["female"])}
            per_gender_actual = min(per_gender, counts["male"], counts["female"])

 
            if per_gender_actual <= 0:
                continue

            random.shuffle(buckets["male"])
            random.shuffle(buckets["female"])

            selected_rows = []
            selected_rows.extend(buckets["male"][:per_gender_actual])
            selected_rows.extend(buckets["female"][:per_gender_actual])

            for r in selected_rows:
                name = r["name"]
                gender = r["gender"]
                occupation = r["occupation"]

                if gender:
                    tr = (name, "has gender", gender)
                    if tr not in seen:
                        seen.add(tr)
                        triples.append(tr)

                if occupation:
                    tr = (name, "has occupation", occupation)
                    if tr not in seen:
                        seen.add(tr)
                        triples.append(tr)

        except Exception as e:
            print(f"Warning: failed to read {csv_path}: {e}")

    return triples


def write_csv(triples, out_csv: Path):
	out_csv.parent.mkdir(parents=True, exist_ok=True)

	with out_csv.open("w", encoding="utf-8", newline="") as fh:
		writer = csv.writer(fh)
		writer.writerow(["subject", "relation", "object"])
		for subj, rel, obj in triples:
			writer.writerow([subj, rel, obj])


def main():
    p = argparse.ArgumentParser(description="Generate dataset (subject,relation,object) from wikidata CSVs")
    p.add_argument("--wikidata-dir", default="wikidata", help="Path to wikidata CSV folder")
    p.add_argument("--out-csv", default="data/dataset_imbalanced_200.csv", help="Output CSV file with columns subject,relation,object")
    args = p.parse_args()

    wikidata_dir = Path(args.wikidata_dir)
    if not wikidata_dir.exists():
        print(f"Error: wikidata folder does not exist: {wikidata_dir}")
        raise SystemExit(1)

    # triples = gather_triples(wikidata_dir)
    triples = gather_triples_limited(wikidata_dir, row_limit=200, seed=42)
    # triples = gather_triples_balanced(wikidata_dir, per_gender=50, seed=42)
    write_csv(triples, Path(args.out_csv))

    print(f"Wrote {len(triples)} triples to {args.out_csv}")


if __name__ == "__main__":
    main()
