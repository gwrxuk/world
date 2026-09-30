// Finite-difference check of WorldModel.trainStep gradients (rho = 1, lr = 0,
// consistency off because finite differences also move its stop-gradient target).
// In float32 expect ~1e-1 relative noise; switching nn.ts/model.ts/env.ts to
// Float64Array gives ~1e-6, which is the meaningful check.
import { collectEpisode, FRAME } from "../lib/wm/env";
import { DEFAULT_CONFIG, WorldModel } from "../lib/wm/model";
import { makeRng } from "../lib/wm/nn";

const cfg = { ...DEFAULT_CONFIG, horizon: 3, batch: 2, lr: 0, rho: 1, wConsistency: 0, hidden: 16, latent: 4 };
const ep = collectEpisode(makeRng(3), cfg.horizon); // one window only => deterministic batch
const m = new WorldModel(cfg);
const J = () => {
  const l = m.trainStep([ep]);
  const n = cfg.horizon + 1;
  return cfg.wRecon * l.recon * n * FRAME + cfg.wReward * l.reward * n + cfg.wConsistency * l.consistency * cfg.horizon * cfg.latent;
};
J();
const params = m.params();
const analytic = params.map((p) => Float32Array.from(p.grad));
const rng = makeRng(9);
let worst = 0;
for (let trial = 0; trial < 40; trial++) {
  const pi = Math.floor(rng() * params.length);
  const p = params[pi];
  const i = Math.floor(rng() * p.value.length);
  const eps = 3e-3;
  const orig = p.value[i];
  p.value[i] = orig + eps; const up = J();
  p.value[i] = orig - eps; const dn = J();
  p.value[i] = orig;
  const num = (up - dn) / (2 * eps);
  const an = analytic[pi][i];
  const rel = Math.abs(num - an) / Math.max(1e-3, Math.abs(num) + Math.abs(an));
  worst = Math.max(worst, rel);
  if (rel > 0.01) console.log(pi, i, num.toFixed(5), an.toFixed(5));
}
console.log(`max relative error over 40 random coordinates: ${worst.toExponential(2)}`);
