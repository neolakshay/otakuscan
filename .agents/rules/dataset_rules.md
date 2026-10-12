# OTAKUSCAN PROJECT RULE: DATASET IS READ-ONLY

This is a permanent project-level constraint.

Original dataset location:
    dataset/

Permitted derived-data location:
    dataset_cpy/

## Strict Rules

1. Treat the entire `dataset/` directory as **READ-ONLY**.
2. Never modify, overwrite, rename, move, delete, or replace any file inside `dataset/`.
3. Never edit:
   - `dataset/sequences.json`
   - `dataset/development/labels.jsonl`
   - `dataset/sample_submission.jsonl`
   - `dataset/score.py`
   - Any original manga image
   - Any other file or directory inside `dataset/`
4. If an operation requires modified data, create or use `dataset_cpy/` instead (e.g., converted images, resized/cropped images, generated annotations, augmented training data, experimental copies).
5. Never silently modify the original dataset for convenience.
6. Read-only operations are allowed:
   - Reading files
   - Parsing JSON and JSONL
   - Inspecting images
   - Calculating statistics
   - Running the official scorer with dataset files as inputs
   - Checking data integrity
7. All experiments must preserve the original source data. If an experiment needs a writable copy, create the necessary files under `dataset_cpy/` and document the transformation.
8. Before any operation that could write to `dataset/`, stop and choose a safe alternative.
9. Do not create `dataset_cpy/` by duplicating the entire dataset unless a concrete experiment requires it. The copy may contain only the files needed for that experiment.
10. `dataset_cpy/` is excluded in `.gitignore`.
11. Do not change the official scoring script (`dataset/score.py`). If custom evaluation code is required, create a separate script outside `dataset/`.
12. Do not generate or manually edit official test predictions while conducting dataset reconnaissance.

## Implementation Requirement

Design all data-processing code to read from `dataset/` and write generated artifacts to appropriate locations outside `dataset/`.

Before modifying any project files, state which paths will be read and which paths will be written.

If a proposed operation could modify `dataset/`, do not execute it.
