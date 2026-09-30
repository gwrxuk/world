// Minimal dependency-free neural network kit: row-major matrices, dense
// layers with explicit caches (so one layer can be applied many times inside
// an unrolled rollout), activations, and Adam.

export interface Mat {
  rows: number;
  cols: number;
  data: Float32Array;
}

export function mat(rows: number, cols: number, data?: Float32Array): Mat {
  return { rows, cols, data: data ?? new Float32Array(rows * cols) };
}

/** Deterministic PRNG (mulberry32) so experiments are reproducible. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export class Param {
  value: Float32Array;
  grad: Float32Array;
  m: Float32Array;
  v: Float32Array;
  constructor(size: number) {
    this.value = new Float32Array(size);
    this.grad = new Float32Array(size);
    this.m = new Float32Array(size);
    this.v = new Float32Array(size);
  }
}

export class Dense {
  W: Param; // [inDim, outDim]
  b: Param; // [outDim]
  constructor(public inDim: number, public outDim: number, rng: () => number, gain = 1) {
    this.W = new Param(inDim * outDim);
    this.b = new Param(outDim);
    const std = gain * Math.sqrt(1 / inDim);
    for (let i = 0; i < this.W.value.length; i++) this.W.value[i] = gaussian(rng) * std;
  }

  forward(x: Mat): Mat {
    const { rows } = x;
    const I = this.inDim;
    const O = this.outDim;
    const y = mat(rows, O);
    const W = this.W.value;
    const b = this.b.value;
    for (let r = 0; r < rows; r++) {
      const yo = r * O;
      for (let o = 0; o < O; o++) y.data[yo + o] = b[o];
      const xo = r * I;
      for (let i = 0; i < I; i++) {
        const xv = x.data[xo + i];
        if (xv === 0) continue;
        const wo = i * O;
        for (let o = 0; o < O; o++) y.data[yo + o] += xv * W[wo + o];
      }
    }
    return y;
  }

  /** Accumulates dW, db from (x, dy) and returns dx. */
  backward(x: Mat, dy: Mat, needDx = true): Mat | null {
    const { rows } = x;
    const I = this.inDim;
    const O = this.outDim;
    const W = this.W.value;
    const gW = this.W.grad;
    const gb = this.b.grad;
    const dx = needDx ? mat(rows, I) : null;
    for (let r = 0; r < rows; r++) {
      const yo = r * O;
      const xo = r * I;
      for (let o = 0; o < O; o++) gb[o] += dy.data[yo + o];
      for (let i = 0; i < I; i++) {
        const xv = x.data[xo + i];
        const wo = i * O;
        let acc = 0;
        for (let o = 0; o < O; o++) {
          const g = dy.data[yo + o];
          gW[wo + o] += xv * g;
          acc += W[wo + o] * g;
        }
        if (dx) dx.data[xo + i] = acc;
      }
    }
    return dx;
  }

  params(): Param[] {
    return [this.W, this.b];
  }
}

export type Act = "tanh" | "relu" | "sigmoid" | "linear";

export function activate(x: Mat, act: Act): Mat {
  if (act === "linear") return x;
  const y = mat(x.rows, x.cols);
  const n = x.data.length;
  for (let i = 0; i < n; i++) {
    const v = x.data[i];
    y.data[i] = act === "tanh" ? Math.tanh(v) : act === "relu" ? (v > 0 ? v : 0) : 1 / (1 + Math.exp(-v));
  }
  return y;
}

/** Gradient through an activation, expressed with its output y. */
export function activateBack(y: Mat, dy: Mat, act: Act): Mat {
  if (act === "linear") return dy;
  const dx = mat(y.rows, y.cols);
  const n = y.data.length;
  for (let i = 0; i < n; i++) {
    const v = y.data[i];
    const d = act === "tanh" ? 1 - v * v : act === "relu" ? (v > 0 ? 1 : 0) : v * (1 - v);
    dx.data[i] = dy.data[i] * d;
  }
  return dx;
}

/** A multi-layer perceptron whose forward pass returns a replayable trace. */
export class MLP {
  layers: Dense[];
  constructor(sizes: number[], public hidden: Act, public out: Act, rng: () => number) {
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) this.layers.push(new Dense(sizes[i], sizes[i + 1], rng));
  }

  forward(x: Mat): { y: Mat; trace: Mat[] } {
    const trace: Mat[] = [x];
    let h = x;
    this.layers.forEach((layer, i) => {
      h = activate(layer.forward(h), i === this.layers.length - 1 ? this.out : this.hidden);
      trace.push(h);
    });
    return { y: h, trace };
  }

  backward(trace: Mat[], dy: Mat, needDx = true): Mat | null {
    let g: Mat | null = dy;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const act = i === this.layers.length - 1 ? this.out : this.hidden;
      g = activateBack(trace[i + 1], g!, act);
      g = this.layers[i].backward(trace[i], g, needDx || i > 0);
    }
    return g;
  }

  params(): Param[] {
    return this.layers.flatMap((l) => l.params());
  }
}

export class Adam {
  t = 0;
  constructor(
    public params: Param[],
    public lr = 1e-3,
    public b1 = 0.9,
    public b2 = 0.999,
    public eps = 1e-8,
    public clip = 1.0,
  ) {}

  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0);
  }

  /** Global-norm gradient clipping followed by an Adam update. Returns the pre-clip norm. */
  step(): number {
    let sq = 0;
    for (const p of this.params) for (let i = 0; i < p.grad.length; i++) sq += p.grad[i] * p.grad[i];
    const norm = Math.sqrt(sq);
    const scale = norm > this.clip ? this.clip / norm : 1;
    this.t++;
    const c1 = 1 - Math.pow(this.b1, this.t);
    const c2 = 1 - Math.pow(this.b2, this.t);
    for (const p of this.params) {
      for (let i = 0; i < p.value.length; i++) {
        const g = p.grad[i] * scale;
        p.m[i] = this.b1 * p.m[i] + (1 - this.b1) * g;
        p.v[i] = this.b2 * p.v[i] + (1 - this.b2) * g * g;
        p.value[i] -= (this.lr * (p.m[i] / c1)) / (Math.sqrt(p.v[i] / c2) + this.eps);
      }
    }
    return norm;
  }
}

export function concatCols(a: Mat, b: Mat): Mat {
  const y = mat(a.rows, a.cols + b.cols);
  for (let r = 0; r < a.rows; r++) {
    y.data.set(a.data.subarray(r * a.cols, (r + 1) * a.cols), r * y.cols);
    y.data.set(b.data.subarray(r * b.cols, (r + 1) * b.cols), r * y.cols + a.cols);
  }
  return y;
}

export function splitCols(x: Mat, left: number): [Mat, Mat] {
  const a = mat(x.rows, left);
  const b = mat(x.rows, x.cols - left);
  for (let r = 0; r < x.rows; r++) {
    a.data.set(x.data.subarray(r * x.cols, r * x.cols + left), r * left);
    b.data.set(x.data.subarray(r * x.cols + left, (r + 1) * x.cols), r * b.cols);
  }
  return [a, b];
}

export function addInto(target: Mat, src: Mat, scale = 1): void {
  for (let i = 0; i < target.data.length; i++) target.data[i] += scale * src.data[i];
}
