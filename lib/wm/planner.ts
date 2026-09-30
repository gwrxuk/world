// Cross-entropy-method (CEM) model-predictive control. The planner is generic
// over a batched scorer, so the same code plans inside NextWM's imagination
// or inside the true simulator (an oracle upper bound).

import { ACT_DIM, EnvState, OBS_DIM, reward, stepEnv } from "./env";
import { WorldModel } from "./model";
import { gaussian, mat } from "./nn";

export interface CEMConfig {
  horizon: number;
  population: number;
  elites: number;
  iterations: number;
}

export const DEFAULT_CEM: CEMConfig = { horizon: 12, population: 200, elites: 20, iterations: 4 };

/** Scores `pop` action sequences, flattened as [pop][horizon][ACT_DIM]. Higher is better. */
export type Scorer = (actions: Float32Array, pop: number, horizon: number) => Float32Array;

export class CEMPlanner {
  private mean: Float32Array;
  constructor(public cfg: CEMConfig, private rng: () => number) {
    this.mean = new Float32Array(cfg.horizon * ACT_DIM);
  }

  reset(): void {
    this.mean.fill(0);
  }

  plan(score: Scorer): [number, number] {
    const { horizon: H, population: P, elites: K, iterations } = this.cfg;
    const D = H * ACT_DIM;
    const mean = this.mean;
    const std = new Float32Array(D).fill(0.8);
    const samples = new Float32Array(P * D);
    for (let it = 0; it < iterations; it++) {
      for (let p = 0; p < P; p++) {
        for (let d = 0; d < D; d++) {
          // Keep one sample at the current mean so the best plan is never lost.
          const v = p === 0 ? mean[d] : mean[d] + std[d] * gaussian(this.rng);
          samples[p * D + d] = Math.max(-1, Math.min(1, v));
        }
      }
      const scores = score(samples, P, H);
      const order = Array.from({ length: P }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
      mean.fill(0);
      std.fill(0);
      for (let e = 0; e < K; e++) for (let d = 0; d < D; d++) mean[d] += samples[order[e] * D + d] / K;
      for (let e = 0; e < K; e++)
        for (let d = 0; d < D; d++) std[d] += (samples[order[e] * D + d] - mean[d]) ** 2 / K;
      for (let d = 0; d < D; d++) std[d] = Math.sqrt(std[d]) + 0.05;
    }
    const first: [number, number] = [mean[0], mean[1]];
    // Warm-start the next call with the plan shifted by one step.
    mean.copyWithin(0, ACT_DIM);
    mean.fill(0, D - ACT_DIM);
    return first;
  }
}

/** Score action sequences by summed predicted reward inside the learned latent space. */
export function modelScorer(model: WorldModel, obs: Float32Array): Scorer {
  const z0 = model.encode(mat(1, OBS_DIM, obs));
  return (actions, pop, H) => {
    const L = z0.cols;
    let z = mat(pop, L);
    for (let p = 0; p < pop; p++) z.data.set(z0.data, p * L);
    const scores = new Float32Array(pop);
    for (let k = 0; k < H; k++) {
      const a = mat(pop, ACT_DIM);
      for (let p = 0; p < pop; p++) {
        a.data[p * ACT_DIM] = actions[(p * H + k) * ACT_DIM];
        a.data[p * ACT_DIM + 1] = actions[(p * H + k) * ACT_DIM + 1];
      }
      z = model.transition(z, a);
      const r = model.predictReward(z);
      for (let p = 0; p < pop; p++) scores[p] += r.data[p];
    }
    return scores;
  };
}

/** Oracle scorer: the same objective evaluated with the true simulator. */
export function oracleScorer(state: EnvState): Scorer {
  return (actions, pop, H) => {
    const scores = new Float32Array(pop);
    for (let p = 0; p < pop; p++) {
      let s = state;
      for (let k = 0; k < H; k++) {
        s = stepEnv(s, actions[(p * H + k) * ACT_DIM], actions[(p * H + k) * ACT_DIM + 1]);
        scores[p] += reward(s);
      }
    }
    return scores;
  };
}
