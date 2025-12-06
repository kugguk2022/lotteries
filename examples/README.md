# Examples

- `euromillions_hello_world.ipynb`: fetch EuroMillions history, compute marginal ball/star frequencies, score synthetic tickets, and plot simple distributions. Run in Jupyter or VS Code notebooks (`pip install -e ".[dev]"` first).

## Interpreting `evaluate_guess`

- `evaluate_guess(draw, guess)` returns `(ball_hits, star_hits)`.
- A simple ticket score prototype is a weighted sum, e.g., `score = ball_hits + 0.5 * star_hits`, or experiment with ROI weights from the prize table.
- Use it to rank candidate tickets quickly; labs (ROI, agent) can plug in richer metrics later.

## Running the notebook

```bash
pip install -e ".[dev]"
python -m ipykernel install --user --name lotteries
jupyter notebook examples/euromillions_hello_world.ipynb
```

All cells run offline after the initial fetch (cache is respected). Plots are small and render quickly.
