# NExtword
uses SML( small language model) to predict new words

Focus this terminal, then:

```sh
uv run --python .venv/bin/python --with mlx-lm python Tests/terminal_test/predictor.py
```

Default model is MiniCPM5-1B (MLX 4-bit, about 0.6 GB). Granite 4.0 H 350M bf16:

```sh
uv run --python .venv/bin/python --with mlx-lm python Tests/terminal_test/predictor.py --model granite
```

SmolLM2:

```sh
uv run --python .venv/bin/python --with mlx-lm python Tests/terminal_test/predictor.py --model smol
```

Edit `input.txt`. Wrap any span in `[[like this]]` to send it for grammar refine; Tab replaces the wrap with the result. Without `[[ ]]`, Tab still applies the last-line rewrite. Esc clears. Ctrl+C quits.

// This project is immature and intended as a weekend Python coding experiment for fun.
