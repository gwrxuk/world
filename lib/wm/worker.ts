/// <reference lib="webworker" />
// Web Worker that owns the replay buffer and the model, so training and
// planning never block the UI thread.

import { Episode, collectEpisode } from "./env";
import { DEFAULT_CONFIG, SerializedModel, WorldModel } from "./model";
import { makeRng } from "./nn";
import { CEMPlanner, modelScorer } from "./planner";

export type WorkerRequest =
  | { type: "collect"; episodes: number }
  | { type: "train"; steps: number }
  | { type: "stop" }
  | { type: "imagine"; id: number; obs: Float32Array; actions: [number, number][] }
  | { type: "act"; id: number; obs: Float32Array; reset: boolean }
  | { type: "load"; model: SerializedModel }
  | { type: "export" };

export type WorkerResponse =
  | { type: "status"; transitions: number; steps: number; params: number }
  | { type: "loss"; step: number; total: number; recon: number; reward: number; consistency: number }
  | { type: "trainDone" }
  | { type: "imagined"; id: number; frames: Float32Array[]; rewards: number[] }
  | { type: "action"; id: number; action: [number, number] }
  | { type: "exported"; model: SerializedModel };

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const rng = makeRng(Date.now() % 100000);
let model = new WorldModel(DEFAULT_CONFIG);
const buffer: Episode[] = [];
// A smaller population than the offline experiments keeps browser control responsive.
const planner = new CEMPlanner({ horizon: 12, population: 120, elites: 12, iterations: 3 }, rng);
let stopRequested = false;

function post(msg: WorkerResponse): void {
  ctx.postMessage(msg);
}

function status(): void {
  post({
    type: "status",
    transitions: buffer.reduce((n, e) => n + e.actions.length, 0),
    steps: model.steps,
    params: model.parameterCount(),
  });
}

async function train(steps: number): Promise<void> {
  stopRequested = false;
  const every = 25;
  let acc = { total: 0, recon: 0, reward: 0, consistency: 0 };
  for (let i = 1; i <= steps && !stopRequested; i++) {
    const l = model.trainStep(buffer);
    acc.total += l.total;
    acc.recon += l.recon;
    acc.reward += l.reward;
    acc.consistency += l.consistency;
    if (i % every === 0) {
      post({
        type: "loss",
        step: model.steps,
        total: acc.total / every,
        recon: acc.recon / every,
        reward: acc.reward / every,
        consistency: acc.consistency / every,
      });
      acc = { total: 0, recon: 0, reward: 0, consistency: 0 };
      status();
      // Yield so a "stop" message can be handled between chunks.
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  post({ type: "trainDone" });
  status();
}

ctx.onmessage = (ev: MessageEvent<WorkerRequest>) => {
  const msg = ev.data;
  switch (msg.type) {
    case "collect":
      for (let i = 0; i < msg.episodes; i++) buffer.push(collectEpisode(rng));
      status();
      break;
    case "train":
      if (buffer.length === 0) for (let i = 0; i < 50; i++) buffer.push(collectEpisode(rng));
      void train(msg.steps);
      break;
    case "stop":
      stopRequested = true;
      break;
    case "imagine": {
      const { frames, rewards } = model.imagine(msg.obs, msg.actions);
      post({ type: "imagined", id: msg.id, frames, rewards });
      break;
    }
    case "act":
      if (msg.reset) planner.reset();
      post({ type: "action", id: msg.id, action: planner.plan(modelScorer(model, msg.obs)) });
      break;
    case "load":
      model = WorldModel.fromJSON(msg.model);
      status();
      break;
    case "export":
      post({ type: "exported", model: model.toJSON() });
      break;
  }
};

status();
