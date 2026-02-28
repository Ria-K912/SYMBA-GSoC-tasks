# SYMBA-GSoC-tasks

Tasks for the SYMBA project (Google Summer of Code): preprocessing SYMBA amplitude data and training models to predict **squared amplitudes** from **amplitudes** (and optionally diagram structure).

---

## Repository structure

| Folder | Description |
|--------|-------------|
| **Task_1.2** | Dataset preprocessing: load SYMBA `.txt` data, normalise indices, tokenise with a regex tokeniser, build vocabulary, 80-10-10 split. |
| **Task_2_vanilla_transformer** | Vanilla encoder-decoder Transformer: amplitude token sequence → squared-amplitude token sequence. Notebooks for QED and QCD. |
| **Task_2_physics_informed_model** | Physics-informed model: dual encoding (text + diagram graph via GNN) and decoder with physics-type token embeddings. Notebooks for QED and QCD. |

---

## Task 1.2 – Preprocessing

- **Input:** SYMBA `.txt` files (per model prefix, e.g. QED, QCD). Each line: `interaction : diagram : amplitude : squared_amplitude`.
- **Steps:** Load raw data, normalise dummy indices (keep physical indices like `s_12`, `p_1`), tokenise expressions with a regex tokeniser, build a shared vocabulary, 80-10-10 train/val/test split.
- **Output:** DataLoaders and `Vocab` for use in Task 2 notebooks. The regex tokeniser keeps physics units as single tokens (e.g. `m_e`, `s_12`, `\sigma_0`) and avoids external dependencies.

**Run:** Open `Task_1.2/preprocess.ipynb`, set `DATA_DIR` to the folder containing the SYMBA test `.txt` files, run all cells.

---

## Task 2 – Vanilla Transformer

- **Idea:** Standard encoder-decoder Transformer with sinusoidal positional encoding. Input = amplitude token sequence; output = squared-amplitude token sequence (next-token prediction, teacher forcing, cross-entropy loss).
- **Notebooks:** `run_task2-QED.ipynb`, `run_task2-QCD.ipynb` in `Task_2_vanilla_transformer/`.
- **Run:** From the repo root (so `preprocess` is importable), open the notebook, set `MODEL` ("QED" or "QCD") and `DATA_DIR`, run all cells. Trained checkpoints are saved under `results_task2_{MODEL}/`.

---

## Task 2 – Physics-informed model

- **Encoding:** (1) Transformer encoder on the amplitude token sequence. (2) Diagram parsed into a PyG graph (vertices, external legs, propagators); GNN (TransformerConv) produces node embeddings. Both are combined into a single memory (concatenated or cross-attention fused).
- **Decoding:** Autoregressive decoder with token embeddings plus **physics-type embeddings** (coupling, mass, Mandelstam, number, regulator, operator, etc.), cross-attending to the fused memory.
- **Notebooks:** `run_task3_QED.ipynb`, `run_task3_QCD.ipynb` in `Task_2_physics_informed_model/`.
- **Run:** From the repo root, open the notebook, set `MODEL` and `DATA_DIR`, run all cells. Requires `torch-geometric`. Checkpoints under `results_task3_{MODEL}/`.

---

## Requirements

- Python 3.x
- PyTorch
- For physics-informed notebooks: `torch-geometric`

Data: SYMBA test `.txt` files (e.g. in a folder such as `SYMBA - Test Data`), with names starting with the model prefix (QED, QCD).

---

## License

See [LICENSE](LICENSE).
