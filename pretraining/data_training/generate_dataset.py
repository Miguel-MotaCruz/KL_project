
from pathlib import Path
import csv
import argparse


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


def gather_triples_limited(wikidata_dir: Path):
	triples = []
	seen = set()

	for csv_path in sorted(wikidata_dir.glob("query_results_*.csv")):
		try:
			with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
				reader = csv.DictReader(fh)

				row_limit = 200      # <-- NEW
				row_count = 0        # <-- NEW

				for row in reader:
					if row_count >= row_limit:   # <-- NEW: stop at 200 rows
						break
					row_count += 1

					name = (row.get("humanLabel") or row.get("name") or "").strip()
					gender = (row.get("genderLabel") or row.get("gender") or "").strip()
					occupation = (row.get("label") or row.get("occupation") or "").strip()

					if not name:
						row_count -= 1
						continue

					gender_clean = gender.lower()

					if gender_clean not in ("male", "female"):
						row_count -= 1
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

					if not occupation or not gender:
						row_count -= 1

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
	p.add_argument("--out-csv", default="data/dataset_limited.csv", help="Output CSV file with columns subject,relation,object")
	args = p.parse_args()

	wikidata_dir = Path(args.wikidata_dir)
	if not wikidata_dir.exists():
		print(f"Error: wikidata folder does not exist: {wikidata_dir}")
		raise SystemExit(1)

	# triples = gather_triples(wikidata_dir)
	triples = gather_triples_limited(wikidata_dir)
	write_csv(triples, Path(args.out_csv))

	print(f"Wrote {len(triples)} triples to {args.out_csv}")


if __name__ == "__main__":
	main()

