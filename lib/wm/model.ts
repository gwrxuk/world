// NextWM: a compact latent world model.
//
//   encoder   z_t      = tanh(E([frame_t, frame_{t-1}]))
//   dynamics  z_{t+1}  = z_t + F([z_t, a_t])             (residual transition)
//   decoder   frame_t ~ D(z_t)                             (sigmoid pixels)
//   reward    r_t      = R(z_t)
//
// Training unrolls F for H steps from a single encoded latent and applies, at
// every step k, (1) pixel reconstruction of the true frame, (2) reward
// regression, and (3) latent consistency against a stop-gradient encoding of
// the true observation. Loss weights decay by rho^k.

import { ACT_DIM, Episode, FRAME, OBS_DIM } from "./env";
import { Adam, MLP, Mat, Param, concatCols, makeRng, mat, splitCols } from "./nn";

export interface WMConfig {
  latent: number;
  hidden: number;
  horizon: number;
  batch: number;
  lr: number;
  wRecon: number;
  wReward: number;
  wConsistency: number;
  rho: number;
  seed: number;
}

export const DEFAULT_CONFIG: WMConfig = {
  latent: 16,
  hidden: 128,
  horizon: 8,
  batch: 32,
  lr: 2e-3,
  wRecon: 1,
  wReward: 10,
  wConsistency: 2,
  rho: 0.9,
  seed: 1,
};

export interface LossReport {
  total: number;
  recon: number;
  reward: number;
  consistency: number;
  gradNorm: number;
}

export class WorldModel {
  enc: MLP;
  dyn: MLP;
  dec: MLP;
  rew: MLP;
  opt: Adam;
  steps = 0;
  private rng: () => number;

  constructor(public cfg: WMConfig = DEFAULT_CONFIG) {
    const rng = makeRng(cfg.seed);
    this.rng = makeRng(cfg.seed + 7919);
    this.enc = new MLP([OBS_DIM, cfg.hidden, cfg.latent], "tanh", "tanh", rng);
    this.dyn = new MLP([cfg.latent + ACT_DIM, cfg.hidden, cfg.latent], "tanh", "linear", rng);
    this.dec = new MLP([cfg.latent, cfg.hidden, FRAME], "relu", "sigmoid", rng);
    this.rew = new MLP([cfg.latent, 64, 1], "tanh", "linear", rng);
    this.opt = new Adam(this.params(), cfg.lr);
  }

  params(): Param[] {
    return [...this.enc.params(), ...this.dyn.params(), ...this.dec.params(), ...this.rew.params()];
  }

  parameterCount(): number {
    return this.params().reduce((n, p) => n + p.value.length, 0);
  }

  encode(obs: Mat): Mat {
    return this.enc.forward(obs).y;
  }

  /** One residual latent transition for a batch of (z, a). */
  transition(z: Mat, a: Mat): Mat {
    const d = this.dyn.forward(concatCols(z, a)).y;
    const out = mat(z.rows, z.cols);
    for (let i = 0; i < out.data.length; i++) out.data[i] = z.data[i] + d.data[i];
    return out;
  }

  decode(z: Mat): Mat {
    return this.dec.forward(z).y;
  }

  predictReward(z: Mat): Mat {
    return this.rew.forward(z).y;
  }

  /** One gradient step on a batch of length-(H+1) windows sampled from the replay buffer. */
  trainStep(episodes: Episode[]): LossReport {
    const { batch: B, horizon: H, latent: L } = this.cfg;
    const obs: Mat[] = [];
    const frames: Mat[] = [];
    const acts: Mat[] = [];
    const rews: Float32Array[] = [];
    for (let k = 0; k <= H; k++) {
      obs.push(mat(B, OBS_DIM));
      frames.push(mat(B, FRAME));
      rews.push(new Float32Array(B));
      if (k < H) acts.push(mat(B, ACT_DIM));
    }
    for (let b = 0; b < B; b++) {
      const ep = episodes[Math.floor(this.rng() * episodes.length)];
      const t0 = Math.floor(this.rng() * (ep.actions.length - H + 1));
      for (let k = 0; k <= H; k++) {
        const o = ep.obs[t0 + k];
        obs[k].data.set(o, b * OBS_DIM);
        frames[k].data.set(o.subarray(0, FRAME), b * FRAME);
        rews[k][b] = ep.rewards[t0 + k];
        if (k < H) acts[k].data.set(ep.actions[t0 + k], b * ACT_DIM);
      }
    }

    this.opt.zeroGrad();
    let lRecon = 0;
    let lReward = 0;
    let lCons = 0;

    // Forward: encode o_0, then unroll the dynamics open-loop.
    const encTrace = this.enc.forward(obs[0]).trace;
    const zs: Mat[] = [encTrace[encTrace.length - 1]];
    const dynTraces: Mat[][] = [];
    for (let k = 1; k <= H; k++) {
      const { y: d, trace } = this.dyn.forward(concatCols(zs[k - 1], acts[k - 1]));
      dynTraces.push(trace);
      const z = mat(B, L);
      for (let i = 0; i < z.data.length; i++) z.data[i] = zs[k - 1].data[i] + d.data[i];
      zs.push(z);
    }

    // Backward, from the last step to the first, carrying dL/dz through the residual chain.
    let carry: Mat | null = null;
    for (let k = H; k >= 0; k--) {
      const w = Math.pow(this.cfg.rho, k) / B;
      const z = zs[k];
      const g = mat(B, L);

      const { y: xhat, trace: decTrace } = this.dec.forward(z);
      const dX = mat(B, FRAME);
      for (let i = 0; i < dX.data.length; i++) {
        const e = xhat.data[i] - frames[k].data[i];
        lRecon += (e * e) / B;
        dX.data[i] = 2 * e * w * this.cfg.wRecon;
      }
      addTo(g, this.dec.backward(decTrace, dX)!);

      const { y: rhat, trace: rewTrace } = this.rew.forward(z);
      const dR = mat(B, 1);
      for (let b = 0; b < B; b++) {
        const e = rhat.data[b] - rews[k][b];
        lReward += (e * e) / B;
        dR.data[b] = 2 * e * w * this.cfg.wReward;
      }
      addTo(g, this.rew.backward(rewTrace, dR)!);

      if (k > 0) {
        const target = this.enc.forward(obs[k]).y; // stop-gradient target
        for (let i = 0; i < g.data.length; i++) {
          const e = z.data[i] - target.data[i];
          lCons += (e * e) / B;
          g.data[i] += 2 * e * w * this.cfg.wConsistency;
        }
      }

      if (carry) addTo(g, carry);

      if (k > 0) {
        const dIn = this.dyn.backward(dynTraces[k - 1], g)!;
        const [dz] = splitCols(dIn, L);
        carry = mat(B, L);
        for (let i = 0; i < carry.data.length; i++) carry.data[i] = g.data[i] + dz.data[i];
      } else {
        this.enc.backward(encTrace, g, false);
      }
    }

    const gradNorm = this.opt.step();
    this.steps++;
    const n = H + 1;
    const recon = lRecon / n / FRAME;
    const rewardL = lReward / n;
    const consistency = H > 0 ? lCons / H / L : 0;
    return {
      total: this.cfg.wRecon * recon + this.cfg.wReward * rewardL + this.cfg.wConsistency * consistency,
      recon,
      reward: rewardL,
      consistency,
      gradNorm,
    };
  }

  /** Open-loop imagination: decode frames for an action sequence starting from one observation. */
  imagine(obs0: Float32Array, actions: [number, number][]): { frames: Float32Array[]; rewards: number[] } {
    let z = this.encode(mat(1, OBS_DIM, obs0));
    const frames = [this.decode(z).data];
    const rewards = [this.predictReward(z).data[0]];
    for (const a of actions) {
      z = this.transition(z, mat(1, ACT_DIM, Float32Array.from(a)));
      frames.push(this.decode(z).data);
      rewards.push(this.predictReward(z).data[0]);
    }
    return { frames, rewards };
  }

  toJSON(): SerializedModel {
    return { cfg: this.cfg, steps: this.steps, params: this.params().map((p) => Array.from(p.value)) };
  }

  static fromJSON(s: SerializedModel): WorldModel {
    const m = new WorldModel(s.cfg);
    m.steps = s.steps;
    m.params().forEach((p, i) => p.value.set(s.params[i]));
    return m;
  }
}

export interface SerializedModel {
  cfg: WMConfig;
  steps: number;
  params: number[][];
}

function addTo(target: Mat, src: Mat): void {
  for (let i = 0; i < target.data.length; i++) target.data[i] += src.data[i];
}
