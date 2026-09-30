"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { EnvState, FRAME, RES, observe, resetEnv, reward, stepEnv } from "@/lib/wm/env";
import { makeRng } from "@/lib/wm/nn";
import type { WorkerRequest, WorkerResponse } from "@/lib/wm/worker";

type LossPoint = { step: number; total: number };
type Pending = (msg: WorkerResponse) => void;

const TICK_MS = 120;

function drawFrame(canvas: HTMLCanvasElement | null, frame: Float32Array | null) {
  if (!canvas) return;
  const g = canvas.getContext("2d");
  if (!g) return;
  const img = g.createImageData(RES, RES);
  for (let i = 0; i < FRAME; i++) {
    const v = frame ? Math.max(0, Math.min(1, frame[i])) : 0;
    img.data[i * 4] = 40 + 200 * v;
    img.data[i * 4 + 1] = 60 + 180 * v;
    img.data[i * 4 + 2] = 90 + 165 * v;
    img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
}

export default function Lab() {
  const worker = useRef<Worker | null>(null);
  const pending = useRef(new Map<number, Pending>());
  const nextId = useRef(1);
  const [status, setStatus] = useState({ transitions: 0, steps: 0, params: 0 });
  const [curve, setCurve] = useState<LossPoint[]>([]);
  const [training, setTraining] = useState(false);
  const [serverBusy, setServerBusy] = useState(false);
  const [driver, setDriver] = useState<"keys" | "agent">("keys");
  const [readout, setReadout] = useState({ real: 0, dreamed: 0, dreamSteps: 0 });

  const rng = useRef(makeRng(Date.now() % 100000));
  const env = useRef<{ s: EnvState; prev: EnvState }>({ s: resetEnv(rng.current), prev: resetEnv(rng.current) });
  const dream = useRef<{ obs0: Float32Array; actions: [number, number][] } | null>(null);
  const keys = useRef(new Set<string>());
  const realCanvas = useRef<HTMLCanvasElement>(null);
  const dreamCanvas = useRef<HTMLCanvasElement>(null);
  const busy = useRef(false);
  const agentFresh = useRef(true);

  const send = useCallback((msg: WorkerRequest) => worker.current?.postMessage(msg), []);

  const request = useCallback(<T extends WorkerResponse>(build: (id: number) => WorkerRequest): Promise<T> => {
    const id = nextId.current++;
    return new Promise<T>((resolve) => {
      pending.current.set(id, resolve as Pending);
      worker.current?.postMessage(build(id));
    });
  }, []);

  useEffect(() => {
    const w = new Worker(new URL("../lib/wm/worker.ts", import.meta.url));
    worker.current = w;
    w.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      const m = ev.data;
      if (m.type === "status") setStatus({ transitions: m.transitions, steps: m.steps, params: m.params });
      else if (m.type === "loss") setCurve((c) => [...c, { step: m.step, total: m.total }]);
      else if (m.type === "trainDone") setTraining(false);
      else if (m.type === "exported") {
        const blob = new Blob([JSON.stringify(m.model)], { type: "application/json" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `nextwm-${m.model.steps}.json`;
        a.click();
      } else if ("id" in m) {
        pending.current.get(m.id)?.(m);
        pending.current.delete(m.id);
      }
    };
    return () => w.terminate();
  }, []);

  useEffect(() => {
    const down = (e: KeyboardEvent) => keys.current.add(e.key.toLowerCase());
    const up = (e: KeyboardEvent) => keys.current.delete(e.key.toLowerCase());
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  const startDream = useCallback(() => {
    const { s, prev } = env.current;
    dream.current = { obs0: observe(s, prev), actions: [] };
  }, []);

  // Simulation loop: step reality, then re-imagine the whole open-loop dream from its start.
  useEffect(() => {
    const timer = setInterval(async () => {
      if (busy.current) return;
      busy.current = true;
      try {
        const { s, prev } = env.current;
        let a: [number, number];
        if (driver === "agent") {
          const res = await request<Extract<WorkerResponse, { type: "action" }>>((id) => ({
            type: "act",
            id,
            obs: observe(s, prev),
            reset: agentFresh.current,
          }));
          agentFresh.current = false;
          a = res.action;
        } else {
          const k = keys.current;
          a = [
            (k.has("arrowright") || k.has("d") ? 1 : 0) - (k.has("arrowleft") || k.has("a") ? 1 : 0),
            (k.has("arrowdown") || k.has("s") ? 1 : 0) - (k.has("arrowup") || k.has("w") ? 1 : 0),
          ];
        }
        const next = stepEnv(s, a[0], a[1]);
        env.current = { s: next, prev: s };
        const obs = observe(next, s);
        drawFrame(realCanvas.current, obs.subarray(0, FRAME));

        let dreamed = 0;
        if (dream.current) {
          dream.current.actions.push(a);
          const d = dream.current;
          const res = await request<Extract<WorkerResponse, { type: "imagined" }>>((id) => ({
            type: "imagine",
            id,
            obs: d.obs0,
            actions: d.actions,
          }));
          drawFrame(dreamCanvas.current, res.frames[res.frames.length - 1]);
          dreamed = res.rewards[res.rewards.length - 1];
        }
        setReadout({ real: reward(next), dreamed, dreamSteps: dream.current?.actions.length ?? 0 });
      } finally {
        busy.current = false;
      }
    }, TICK_MS);
    return () => clearInterval(timer);
  }, [driver, request]);

  const newEpisode = () => {
    const s = resetEnv(rng.current);
    env.current = { s, prev: s };
    agentFresh.current = true;
    if (dream.current) startDream();
  };

  const trainOnServer = async () => {
    setServerBusy(true);
    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ steps: 1000, episodes: 150 }),
      });
      const json = await res.json();
      send({ type: "load", model: json.model });
      setCurve(json.curve);
    } finally {
      setServerBusy(false);
    }
  };

  const maxLoss = Math.max(1e-6, ...curve.map((p) => p.total));
  const path = curve
    .map((p, i) => `${i === 0 ? "M" : "L"}${(i / Math.max(1, curve.length - 1)) * 300},${130 - (p.total / maxLoss) * 120}`)
    .join(" ");

  return (
    <main>
      <h1>NextWM</h1>
      <p className="sub">
        A latent world model that learns a pixel environment in your browser, dreams its future, and plans inside the dream.
      </p>

      <div className="grid">
        <section className="panel">
          <h2>1. Data and training</h2>
          <div className="row">
            <button onClick={() => send({ type: "collect", episodes: 50 })}>Collect 50 episodes</button>
            {training ? (
              <button onClick={() => send({ type: "stop" })}>Stop</button>
            ) : (
              <button
                className="primary"
                onClick={() => {
                  setTraining(true);
                  send({ type: "train", steps: 1500 });
                }}
              >
                Train 1,500 steps
              </button>
            )}
          </div>
          <div className="row">
            <button disabled={serverBusy || training} onClick={trainOnServer}>
              {serverBusy ? "Training on server…" : "Train on server (API route)"}
            </button>
            <button onClick={() => send({ type: "export" })}>Export weights</button>
          </div>
          <p className="stat">
            {status.transitions.toLocaleString()} transitions · {status.steps.toLocaleString()} gradient steps ·{" "}
            {status.params.toLocaleString()} parameters
          </p>
          <svg className="curve" viewBox="0 0 300 140" preserveAspectRatio="none" aria-label="Training loss">
            <path d={path} fill="none" stroke="var(--accent)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
          </svg>
          <p className="stat">Training loss{curve.length ? `: ${curve[curve.length - 1].total.toFixed(4)}` : ""}</p>
        </section>

        <section className="panel">
          <h2>2. Reality and imagination</h2>
          <div className="row">
            <button className={driver === "keys" ? "primary" : ""} onClick={() => setDriver("keys")}>
              Drive (arrows / WASD)
            </button>
            <button
              className={driver === "agent" ? "primary" : ""}
              onClick={() => {
                agentFresh.current = true;
                setDriver("agent");
              }}
            >
              CEM agent in the dream
            </button>
            <button onClick={newEpisode}>New episode</button>
          </div>
          <div className="row">
            <button onClick={startDream}>{dream.current ? "Re-sync dream" : "Start dreaming"}</button>
            <button onClick={() => (dream.current = null)}>Stop dreaming</button>
          </div>
          <div className="frames">
            <div>
              <div className="label">Reality · reward {readout.real.toFixed(2)}</div>
              <canvas ref={realCanvas} width={RES} height={RES} />
            </div>
            <div>
              <div className="label">
                Open-loop dream · {readout.dreamSteps} steps · predicted reward {readout.dreamed.toFixed(2)}
              </div>
              <canvas ref={dreamCanvas} width={RES} height={RES} />
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
