import { EPISODE_LEN, Episode, FRAME, SUCCESS_RADIUS, observe, resetEnv, reward, stepEnv } from "./env";
import { WorldModel } from "./model";
import { CEMConfig, CEMPlanner, modelScorer, oracleScorer } from "./planner";
import { makeRng } from "./nn";

/**
 * Open-loop prediction error: encode o_t once, roll the dynamics k steps with
 * the logged actions, decode, and compare with the true frame t+k. Also
 * reports the "copy the last frame" baseline at the same horizon.
 */
export function predictionError(model: WorldModel, episodes: Episode[], maxH: number) {
  const mse = new Array(maxH + 1).fill(0);
  const copy = new Array(maxH + 1).fill(0);
  let n = 0;
  for (const ep of episodes) {
    for (let t0 = 0; t0 + maxH < ep.obs.length; t0 += 10) {
      const { frames } = model.imagine(ep.obs[t0], ep.actions.slice(t0, t0 + maxH));
      for (let k = 0; k <= maxH; k++) {
        const truth = ep.obs[t0 + k];
        const last = ep.obs[t0];
        let e = 0;
        let c = 0;
        for (let i = 0; i < FRAME; i++) {
          e += (frames[k][i] - truth[i]) ** 2;
          c += (last[i] - truth[i]) ** 2;
        }
        mse[k] += e / FRAME;
        copy[k] += c / FRAME;
      }
      n++;
    }
  }
  return { model: mse.map((v) => v / n), copyLast: copy.map((v) => v / n), windows: n };
}

export type PolicyKind = "random" | "nextwm" | "oracle";

export interface ControlResult {
  policy: PolicyKind;
  episodes: number;
  meanReturn: number;
  successRate: number;
  meanFinalDistance: number;
}

export function evaluateControl(
  kind: PolicyKind,
  episodes: number,
  seed: number,
  cem: CEMConfig,
  model?: WorldModel,
  len = EPISODE_LEN,
): ControlResult {
  const rng = makeRng(seed);
  const planRng = makeRng(seed + 101);
  const planner = new CEMPlanner(cem, planRng);
  let totalReturn = 0;
  let successes = 0;
  let totalDist = 0;
  for (let e = 0; e < episodes; e++) {
    let s = resetEnv(rng);
    let prev = s;
    planner.reset();
    let ret = 0;
    for (let t = 0; t < len; t++) {
      let a: [number, number];
      if (kind === "random") a = [rng() * 2 - 1, rng() * 2 - 1];
      else if (kind === "oracle") a = planner.plan(oracleScorer(s));
      else a = planner.plan(modelScorer(model!, observe(s, prev)));
      prev = s;
      s = stepEnv(s, a[0], a[1]);
      ret += reward(s);
    }
    const d = -reward(s);
    totalReturn += ret;
    totalDist += d;
    if (d < SUCCESS_RADIUS) successes++;
  }
  return {
    policy: kind,
    episodes,
    meanReturn: totalReturn / episodes,
    successRate: successes / episodes,
    meanFinalDistance: totalDist / episodes,
  };
}

function ranks(x: Float32Array): number[] {
  const idx = Array.from(x, (_, i) => i).sort((a, b) => x[a] - x[b]);
  const r = new Array<number>(x.length);
  idx.forEach((i, k) => (r[i] = k));
  return r;
}

export function spearman(a: Float32Array, b: Float32Array): number {
  const ra = ranks(a);
  const rb = ranks(b);
  const m = (a.length - 1) / 2;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i++) {
    num += (ra[i] - m) * (rb[i] - m);
    da += (ra[i] - m) ** 2;
    db += (rb[i] - m) ** 2;
  }
  return num / Math.sqrt(da * db);
}

/**
 * Planning relevance: mean Spearman correlation between the model's imagined
 * H-step return and the true return over `pop` constant-direction action
 * sequences (the kind CEM converges to), from `states` random start states.
 */
export function rankingCorrelation(model: WorldModel, states = 20, pop = 200, H = 12, seed = 99): number {
  const rng = makeRng(seed);
  let sum = 0;
  for (let n = 0; n < states; n++) {
    let s = resetEnv(rng);
    let prev = s;
    for (let t = 0; t < 5; t++) {
      prev = s;
      s = stepEnv(s, rng() * 2 - 1, rng() * 2 - 1);
    }
    const acts = new Float32Array(pop * H * 2);
    for (let p = 0; p < pop; p++) {
      const ax = rng() * 2 - 1;
      const ay = rng() * 2 - 1;
      for (let k = 0; k < H; k++) {
        acts[(p * H + k) * 2] = ax;
        acts[(p * H + k) * 2 + 1] = ay;
      }
    }
    sum += spearman(modelScorer(model, observe(s, prev))(acts, pop, H), oracleScorer(s)(acts, pop, H));
  }
  return sum / states;
}
