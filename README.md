# Dots and Boxes — play the neural-net + MCTS bot in your browser

A static web app: play 3×3 Dots and Boxes against the trained network guided by
Monte Carlo Tree Search. **Everything runs client-side** — the game rules, the
network's forward pass, and the MCTS are all in plain JavaScript, with the
trained weights shipped as `model.json`. No server, no build step, no
dependencies.

## Files

| file | what it is |
|------|------------|
| `index.html` | the page (board + controls, inline CSS) |
| `dab.js` | game rules + network forward pass + net-guided MCTS |
| `app.js` | UI: renders the board, handles clicks, drives the bot |
| `model.js` | the trained weights, embedded (the seed-1 model, ~0.84 vs the safe heuristic) |

The bot is that network plus **PUCT MCTS** (default 400 simulations per move),
verified to match the Python implementation to float precision. Weights are
embedded as a script (not fetched), so the page works even opened directly as a
local file — no server needed.

## Publish on GitHub Pages

1. Create a new GitHub repository.
2. Copy these four files (`index.html`, `dab.js`, `app.js`, `model.js`) into
   the **repository root** and push.
3. On GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a
   branch**, pick your branch (e.g. `main`) and folder **`/ (root)`**, then Save.
4. Wait ~1 minute. Your site is live at
   `https://<your-username>.github.io/<repo-name>/`.

(If you'd rather keep these in a `docs/` subfolder, put them there instead and
choose the `/docs` folder in the Pages settings.)

## Run it locally

Just open `index.html` in your browser — double-click it, or drag it into a tab.
The weights are embedded, so no server is required.

## Playing tips

- Complete a box to claim it **and move again** (the extra-turn rule).
- The **second player has a structural advantage** on 3×3, so "You play: second"
  is the easier way to win.
- The bot plays the **double-cross** (declining boxes to control the chains). To
  beat it, you'll need to out-manage the endgame chains yourself.
- "Strength" sets the MCTS simulations per move: 100 (easy) → 400 (strong) →
  800 (max). More simulations = stronger and a touch slower.
