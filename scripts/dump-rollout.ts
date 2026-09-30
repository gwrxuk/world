// Dumps one real vs imagined open-loop rollout for the paper's qualitative figure.
//   npx tsx scripts/dump-rollout.ts  ->  results/rollout.json
import { readFileSync, writeFileSync } from "node:fs";
import { FRAME, collectEpisode } from "../lib/wm/env";
import { WorldModel } from "../lib/wm/model";
import { makeRng } from "../lib/wm/nn";

const model = WorldModel.fromJSON(JSON.parse(readFileSync("results/results-model.json", "utf8")));
const ep = collectEpisode(makeRng(4242));
const t0 = 10;
const H = 15;
const { frames } = model.imagine(ep.obs[t0], ep.actions.slice(t0, t0 + H));
const real = Array.from({ length: H + 1 }, (_, k) => Array.from(ep.obs[t0 + k].subarray(0, FRAME)));
writeFileSync("results/rollout.json", JSON.stringify({ real, imagined: frames.map((f) => Array.from(f)) }));
