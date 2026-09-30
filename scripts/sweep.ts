// Quick ablation: train with config overrides, report prediction and ranking quality.
//   npx tsx scripts/sweep.ts <name> <steps> '<json overrides>'
import { mkdirSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { Exploration, collectEpisode } from "../lib/wm/env";
import { DEFAULT_CONFIG, WorldModel } from "../lib/wm/model";
import { makeRng } from "../lib/wm/nn";

const [name, stepsArg, json] = process.argv.slice(2);
const cfg = { ...DEFAULT_CONFIG, ...JSON.parse(json ?? "{}") };
const rng = makeRng(42);
const explore = (process.env.EXPLORE ?? "hold") as Exploration;
const train = Array.from({ length: 200 }, () => collectEpisode(rng, 50, explore));
const m = new WorldModel(cfg);
const t0 = Date.now();
let recon = 0, rew = 0;
for (let i = 1; i <= Number(stepsArg); i++) {
  const l = m.trainStep(train);
  if (i > Number(stepsArg) - 200) { recon += l.recon / 200; rew += l.reward / 200; }
}
mkdirSync("results/sweep", { recursive: true });
const file = `results/sweep/${name}.json`;
writeFileSync(file, JSON.stringify(m.toJSON()));
const diag = execFileSync("npx", ["tsx", "scripts/diagnose.ts", file]).toString().trim();
console.log(JSON.stringify({ name, explore, cfg, recon, reward: rew, seconds: (Date.now() - t0) / 1000, diag }));
