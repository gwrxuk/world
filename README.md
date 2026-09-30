# NextWM

A small latent world model that learns a pixel environment, imagines its future, and plans inside the imagination. It runs entirely on the web platform. The model is plain TypeScript with no ML dependencies. The same code trains in a browser Web Worker and in a Next.js route handler.

## Run it

```bash
npm install
npm run dev            # http://localhost:3000
```

In the lab page:

1. Click **Train 1,500 steps**. It collects exploration data automatically, and the loss streams in live.
2. Click **Start dreaming**, then drive with the arrow keys or WASD. The right canvas shows the model's open-loop imagination of the same actions.
3. Switch to **CEM agent in the dream** to let the planner steer the real environment using only the model.

**Train on server** calls `POST /api/train` with `{ "steps": 1000, "episodes": 150 }`. It trains on Node.js and loads the weights into the browser worker.

## Reproduce the experiments

```bash
npm run experiment -- 6000 hold '{}' results           # full NextWM
npm run experiment -- 6000 ou '{"wReward":1,"lr":0.001}' ou-base   # ablations: ou-full, hold-base
npx tsx scripts/dump-rollout.ts                         # qualitative rollout
python3 scripts/figures.py                              # figures (docs/figures/*.png)
```

Other scripts:

- `scripts/gradcheck.ts`: finite-difference gradient check.
- `scripts/diagnose.ts`: return-ranking correlation of a saved model.
- `scripts/sweep.ts`: quick config sweeps.

## Layout

| Path | Role |
| --- | --- |
| `lib/wm/nn.ts` | Matrices, dense layers, activations, Adam, seeded PRNG |
| `lib/wm/env.ts` | PointNav simulator, renderer, exploration policies |
| `lib/wm/model.ts` | Encoder, residual latent dynamics, decoder, reward head, unrolled training |
| `lib/wm/planner.ts` | CEM planner with model and oracle scorers |
| `lib/wm/evaluate.ts` | Prediction error, return-ranking correlation, control evaluation |
| `lib/wm/worker.ts` | Web Worker owning data and model |
| `app/api/train/route.ts` | Server-side training route |
| `components/Lab.tsx` | Interactive lab UI |
