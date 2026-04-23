from pathlib import Path

import pandas as pd


def read_table(file_path: Path) -> pd.DataFrame:
	suffix = file_path.suffix.lower()

	if suffix == ".csv":
		return pd.read_csv(file_path)
	if suffix == ".tsv":
		return pd.read_csv(file_path, sep="\t")
	if suffix in {".xlsx", ".xls"}:
		# Read first sheet from each Excel file
		return pd.read_excel(file_path)

	raise ValueError(f"Unsupported file type: {file_path.name}")


def main() -> None:
	folder = Path(__file__).resolve().parent
	output_file = folder / "merged_tables.xlsx"

	supported = {".csv", ".tsv", ".xlsx", ".xls"}
	input_files = sorted(
		[
			f
			for f in folder.iterdir()
			if f.is_file() and f.suffix.lower() in supported and f.name != output_file.name
		]
	)

	if not input_files:
		print("No table files found to merge.")
		return

	merged_frames = []
	with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
		for file_path in input_files:
			try:
				df = read_table(file_path)
				df.insert(0, "source_file", file_path.name)
				merged_frames.append(df)

				sheet_name = file_path.stem[:31] or "Sheet"
				df.to_excel(writer, sheet_name=sheet_name, index=False)
			except Exception as e:
				print(f"Skipping {file_path.name}: {e}")

		if merged_frames:
			merged_df = pd.concat(merged_frames, ignore_index=True, sort=False)
			merged_df.to_excel(writer, sheet_name="merged", index=False)

	print(f"Workbook created: {output_file}")


if __name__ == "__main__":
	main()
