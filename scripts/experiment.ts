// Reproduces every number reported in the paper.
//   npm run experiment -- [trainSteps] [hold|ou] ['{"wReward":10}'] [outName]
// Writes results/<outName>.json (default results/results.json) and the trained weights.

import { mkdirSync, writeFileSync } from "node:fs";
import { Exploration, collectEpisode } from "../lib/wm/env";
import { evaluateControl, predictionError, rankingCorrelation } from "../lib/wm/evaluate";
import { DEFAULT_CONFIG, WorldModel } from "../lib/wm/model";
import { makeRng } from "../lib/wm/nn";
import { DEFAULT_CEM } from "../lib/wm/planner";

const TRAIN_STEPS = Number(process.argv[2] ?? 4000);
const EXPLORE = (process.argv[3] ?? "hold") as Exploration;
const CONFIG = { ...DEFAULT_CONFIG, ...JSON.parse(process.argv[4] ?? "{}") };
const OUT = process.argv[5] ?? "results";
const N_TRAIN_EPISODES = 200;
const N_TEST_EPISODES = 40;
const N_CONTROL_EPISODES = 30;
const PRED_H = 15;

const dataRng = makeRng(42);
const train = Array.from({ length: N_TRAIN_EPISODES }, () => collectEpisode(dataRng, 50, EXPLORE));
const testRng = makeRng(4242);
const test = Array.from({ length: N_TEST_EPISODES }, () => collectEpisode(testRng));

const model = new WorldModel(CONFIG);
console.log(`params=${model.parameterCount()} transitions=${N_TRAIN_EPISODES * 50}`);

const curve: { step: number; total: number; recon: number; reward: number; consistency: number }[] = [];
const t0 = Date.now();
let acc = { total: 0, recon: 0, reward: 0, consistency: 0 };
for (let step = 1; step <= TRAIN_STEPS; step++) {
  const l = model.trainStep(train);
  acc.total += l.total;
  acc.recon += l.recon;
  acc.reward += l.reward;
  acc.consistency += l.consistency;
  if (step % 100 === 0) {
    const row = {
      step,
      total: acc.total / 100,
      recon: acc.recon / 100,
      reward: acc.reward / 100,
      consistency: acc.consistency / 100,
    };
    curve.push(row);
    acc = { total: 0, recon: 0, reward: 0, consistency: 0 };
    console.log(JSON.stringify(row));
  }
}
const trainSeconds = (Date.now() - t0) / 1000;
console.log(`trained ${TRAIN_STEPS} steps in ${trainSeconds.toFixed(1)}s`);

const pred = predictionError(model, test, PRED_H);
console.log("prediction", JSON.stringify(pred));

const ranking = rankingCorrelation(model);
console.log("ranking", ranking);

const control = (["random", "nextwm", "oracle"] as const).map((k) => {
  const t = Date.now();
  const r = evaluateControl(k, N_CONTROL_EPISODES, 777, DEFAULT_CEM, model);
  const res = { ...r, secondsPerEpisode: (Date.now() - t) / 1000 / N_CONTROL_EPISODES };
  console.log(JSON.stringify(res));
  return res;
});

mkdirSync("results", { recursive: true });
writeFileSync(
  `results/${OUT}.json`,
  JSON.stringify(
    {
      config: CONFIG,
      exploration: EXPLORE,
      ranking,
      cem: DEFAULT_CEM,
      parameters: model.parameterCount(),
      trainEpisodes: N_TRAIN_EPISODES,
      trainSteps: TRAIN_STEPS,
      trainSeconds,
      node: process.version,
      curve,
      prediction: pred,
      control,
    },
    null,
    2,
  ),
);
writeFileSync(`results/${OUT}-model.json`, JSON.stringify(model.toJSON()));
