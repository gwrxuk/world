// PointNav: a point mass with momentum must reach a goal. The agent only sees
// pixels: two stacked 16x16 grayscale frames (agent blob = 1.0, goal blob = 0.5).
// Velocity is not visible in a single frame, which is why frames are stacked.

export const RES = 16;
export const FRAME = RES * RES;
export const OBS_DIM = 2 * FRAME;
export const ACT_DIM = 2;
export const EPISODE_LEN = 50;

const DAMPING = 0.8;
const GAIN = 0.05;
const BLOB_SIGMA = 0.15;
export const SUCCESS_RADIUS = 0.2;

export interface EnvState {
  x: number;
  y: number;
  vx: number;
  vy: number;
  gx: number;
  gy: number;
  t: number;
}

function clampUnit(v: number): number {
  return Math.max(-1, Math.min(1, v));
}

export function resetEnv(rng: () => number): EnvState {
  const u = () => rng() * 1.6 - 0.8;
  return { x: u(), y: u(), vx: 0, vy: 0, gx: u(), gy: u(), t: 0 };
}

export function reward(s: EnvState): number {
  return -Math.hypot(s.x - s.gx, s.y - s.gy);
}

/** Ground-truth transition. Walls absorb momentum along the blocked axis. */
export function stepEnv(s: EnvState, ax: number, ay: number): EnvState {
  let vx = DAMPING * s.vx + GAIN * clampUnit(ax);
  let vy = DAMPING * s.vy + GAIN * clampUnit(ay);
  let x = s.x + vx;
  let y = s.y + vy;
  if (x < -1 || x > 1) {
    x = clampUnit(x);
    vx = 0;
  }
  if (y < -1 || y > 1) {
    y = clampUnit(y);
    vy = 0;
  }
  return { x, y, vx, vy, gx: s.gx, gy: s.gy, t: s.t + 1 };
}

export function renderFrame(s: EnvState, out: Float32Array = new Float32Array(FRAME), offset = 0): Float32Array {
  const k = 1 / (2 * BLOB_SIGMA * BLOB_SIGMA);
  for (let j = 0; j < RES; j++) {
    const py = -1 + ((j + 0.5) * 2) / RES;
    for (let i = 0; i < RES; i++) {
      const px = -1 + ((i + 0.5) * 2) / RES;
      const a = Math.exp(-((px - s.x) ** 2 + (py - s.y) ** 2) * k);
      const g = 0.5 * Math.exp(-((px - s.gx) ** 2 + (py - s.gy) ** 2) * k);
      out[offset + j * RES + i] = Math.min(1, a + g);
    }
  }
  return out;
}

/** Observation = [current frame, previous frame]. */
export function observe(s: EnvState, prev: EnvState): Float32Array {
  const o = new Float32Array(OBS_DIM);
  renderFrame(s, o, 0);
  renderFrame(prev, o, FRAME);
  return o;
}

export interface Transition {
  obs: Float32Array;
  action: [number, number];
  reward: number; // reward of the state reached after the action
  nextObs: Float32Array;
}

export interface Episode {
  obs: Float32Array[]; // length T+1
  actions: [number, number][]; // length T
  rewards: number[]; // length T+1, rewards[t] = r(s_t)
}

export type Exploration = "hold" | "ou";

/**
 * Exploration policy for data collection.
 * - "hold" (default): a uniformly random action held for 1-10 steps, so the data
 *   contains the sustained full-force moves a planner will choose.
 * - "ou": temporally correlated noise (Ornstein-Uhlenbeck-like). Its actions
 *   mostly cancel out, so high velocities are rare in the data.
 */
export function collectEpisode(rng: () => number, len = EPISODE_LEN, mode: Exploration = "hold"): Episode {
  let s = resetEnv(rng);
  let prev = s;
  const obs = [observe(s, prev)];
  const rewards = [reward(s)];
  const actions: [number, number][] = [];
  let ax = 0;
  let ay = 0;
  let hold = 0;
  for (let t = 0; t < len; t++) {
    if (mode === "ou") {
      ax = clampUnit(0.7 * ax + 0.6 * (rng() * 2 - 1));
      ay = clampUnit(0.7 * ay + 0.6 * (rng() * 2 - 1));
    } else if (hold-- <= 0) {
      ax = rng() * 2 - 1;
      ay = rng() * 2 - 1;
      hold = Math.floor(rng() * 10);
    }
    const next = stepEnv(s, ax, ay);
    prev = s;
    s = next;
    actions.push([ax, ay]);
    obs.push(observe(s, prev));
    rewards.push(reward(s));
  }
  return { obs, actions, rewards };
}
