# Puzzle generation

Run the generator from the repository root with the `gfpack` environment active:

```bash
python scripts/genPuzzle.py --rect 1 --cut 15 --datasize 16 \
  --seed 3407 --output data/datasets/puzzle_rect.pkl
```

This produces 16 samples with 16 pieces each inside a 2000 x 2000 rectangle.
Rectangle mode does not require a boundary file. `--cut` counts cuts, so each
sample contains `cut + 1` pieces. Visualizations go to `puzzle_vis/`; use
`--vis_dir PATH` to change this or `--vis_count 0` to disable visualization.

For the generator's existing polygon-boundary mode, supply a JSON file or a
trusted pickle containing a list of boundary rings. For example, `boundary.json`:

```json
[
  [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]]
]
```

```bash
python scripts/genPuzzle.py --boundary boundary.json --cut 15 --datasize 16 \
  --output data/datasets/puzzle_custom.pkl
```

Each boundary must be a simple polygon with positive area and finite N x 2
vertices. Either winding and an optional closing vertex are accepted. Holes are
not supported. Each boundary is translated to minimum x/y zero and uniformly
scaled to a longest bounding-box side of 2000. The default input is
`boundary.pkl` when `--rect 0` is used.

`--seed` controls boundary selection, cut paths, and rotations. Generation is
bounded by `--max_path_steps` (2048 steps per path) and `--max_cut_attempts`
(256 trials per accepted cut). If generation cannot complete, it exits with an
error before writing the dataset. Try a different seed or step length for
particularly narrow or complex boundaries.

The output preserves the existing format: three consecutive pickle objects:

```python
import pickle

with open("data/datasets/puzzle_rect.pkl", "rb") as file:
    vertices = pickle.load(file)      # samples -> pieces -> local [x, y] vertices
    poses = pickle.load(file)         # samples -> pieces -> poses
    boundary_ids = pickle.load(file)  # one boundary index per sample
```

Poses are `[tx, ty, cos(theta), sin(theta)]` by default, or `[tx, ty]` with
`--rotation 0`. Recover world coordinates with `R(theta) @ local + [tx, ty]`.
Boundary IDs index the input boundary list; `-1` denotes the built-in rectangle.
Keep the boundary input file to resolve the IDs. This is the puzzle generator's
format, separate from the Dental loaders' polygon-ID format.

Run the regression checks with:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```
