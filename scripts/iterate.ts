// One round of the model-based RL data loop: collect episodes with the CEM
// planner acting inside the current model (plus exploration noise), add them to
// the original exploration data, and continue training.
//   npx tsx scripts/iterate.ts <init> <plannerEpisodes> <steps> <out>
import { readFileSync, writeFileSync } from "node:fs";
import { Episode, collectEpisode, observe, resetEnv, reward, stepEnv } from "../lib/wm/env";
import { evaluateControl, predictionError, rankingCorrelation } from "../lib/wm/evaluate";
import { WorldModel } from "../lib/wm/model";
import { gaussian, makeRng } from "../lib/wm/nn";
import { CEMPlanner, DEFAULT_CEM, modelScorer } from "../lib/wm/planner";

const [init = "results", nArg = "100", stepsArg = "3000", out = "iter1"] = process.argv.slice(2);
const prior = JSON.parse(readFileSync(`results/${init}.json`, "utf8"));
const model = WorldModel.fromJSON(JSON.parse(readFileSync(`results/${init}-model.json`, "utf8")));

const dataRng = makeRng(42);
const data: Episode[] = Array.from({ length: prior.trainEpisodes }, () => collectEpisode(dataRng, 50, prior.exploration));

const rng = makeRng(2024);
const planner = new CEMPlanner({ ...DEFAULT_CEM, population: 100, iterations: 3 }, makeRng(7));
const t0 = Date.now();
for (let e = 0; e < Number(nArg); e++) {
  let s = resetEnv(rng);
  let prev = s;
  planner.reset();
  const ep: Episode = { obs: [observe(s, prev)], actions: [], rewards: [reward(s)] };
  for (let t = 0; t < 50; t++) {
    const a = planner.plan(modelScorer(model, ep.obs[t]));
    const act: [number, number] = [
      Math.max(-1, Math.min(1, a[0] + 0.3 * gaussian(rng))),
      Math.max(-1, Math.min(1, a[1] + 0.3 * gaussian(rng))),
    ];
    prev = s;
    s = stepEnv(s, act[0], act[1]);
    ep.actions.push(act);
    ep.obs.push(observe(s, prev));
    ep.rewards.push(reward(s));
  }
  data.push(ep);
}
const collectSeconds = (Date.now() - t0) / 1000;
console.log(`collected ${nArg} planner episodes in ${collectSeconds.toFixed(0)}s`);

const curve: { step: number; total: number; recon: number; reward: number; consistency: number }[] = [];
const t1 = Date.now();
let acc = { total: 0, recon: 0, reward: 0, consistency: 0 };
for (let step = 1; step <= Number(stepsArg); step++) {
  const l = model.trainStep(data);
  acc.total += l.total;
  acc.recon += l.recon;
  acc.reward += l.reward;
  acc.consistency += l.consistency;
  if (step % 100 === 0) {
    curve.push({ step: prior.trainSteps + step, ...Object.fromEntries(Object.entries(acc).map(([k, v]) => [k, v / 100])) } as never);
    acc = { total: 0, recon: 0, reward: 0, consistency: 0 };
  }
}
const trainSeconds = (Date.now() - t1) / 1000;

const testRng = makeRng(4242);
const test = Array.from({ length: 40 }, () => collectEpisode(testRng));
const prediction = predictionError(model, test, 15);
const ranking = rankingCorrelation(model);
console.log("ranking", ranking);
const control = (["random", "nextwm", "oracle"] as const).map((k) => {
  const t = Date.now();
  const r = evaluateControl(k, 30, 777, DEFAULT_CEM, model);
  const res = { ...r, secondsPerEpisode: (Date.now() - t) / 1000 / 30 };
  console.log(JSON.stringify(res));
  return res;
});
writeFileSync(
  `results/${out}.json`,
  JSON.stringify(
    {
      ...prior,
      init,
      plannerEpisodes: Number(nArg),
      extraSteps: Number(stepsArg),
      collectSeconds,
      extraTrainSeconds: trainSeconds,
      curve: [...prior.curve, ...curve],
      prediction,
      ranking,
      control,
    },
    null,
    2,
  ),
);
writeFileSync(`results/${out}-model.json`, JSON.stringify(model.toJSON()));
