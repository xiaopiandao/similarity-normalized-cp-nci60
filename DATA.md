# Data acquisition and layout

## Source

The experiments use public classic NCI-60 and DTP HTS384 exports from the NCI CellMiner download portal:

https://discover.nci.nih.gov/cellminer/loadDownload.do

Download and redistribution remain subject to the source portal's current terms. Raw source files are not included in this GitHub package.

## Classic NCI-60 layout

After extracting the CellMiner downloads, place the files as follows:

```text
data/raw/cellminer/extracted/
├── DTP_NCI60_RAW/output/DTP_NCI60_RAW.xlsx
├── DTP_NCI60_ZSCORE/output/DTP_NCI60_ZSCORE.xlsx
├── nci60_RNA__RNA_seq_composite_expression/output/html/RNA__RNA_seq_composite_expression.html
├── nci60_DNA__Combined_aCGH_gene_summary/output/html/DNA__Combined_aCGH_gene_summary.html
└── nci60_DNA__Exome_Seq_Protein_function_affecting/output/html/DNA__Exome_Seq_Protein_function_affecting.html
```

Then run:

```bash
python scripts/preprocess_cellminer.py --root .
python scripts/make_basic_splits.py --root . --seeds 1,2,3,4,5
python scripts/make_chemical_splits.py --root . --seeds 1,2,3,4,5
python scripts/prepare_phase1a.py --root . --seeds 1,2,3,4,5
```

## HTS384 layout

Place the extracted external-assay files at:

```text
data/raw/cellminer/hts384/extracted/
├── DTP_HTS384_NCI60_RAW/output/DTP_HTS384_NCI60_RAW.xlsx
└── DTP_HTS384_NCI60_ZSCORE/output/DTP_HTS384_NCI60_ZSCORE.xlsx
```

Run:

```bash
python scripts/prepare_hts384_lockbox.py --root .
```

This step can query PubChem for structures not already available in the classic collection and writes a local cache under `data/processed/cellminer_hts384/`.

## Files intentionally excluded from Git

The following are ignored: raw downloads, processed response and feature matrices, model checkpoints, generated results, logs, caches, and compressed archives. Keep the source download filenames and record their checksums locally for a complete audit trail.
