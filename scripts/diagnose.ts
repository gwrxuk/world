// Planning diagnostic: Spearman correlation between imagined and true returns.
//   npx tsx scripts/diagnose.ts [model.json]
import { readFileSync } from "node:fs";
import { rankingCorrelation } from "../lib/wm/evaluate";
import { WorldModel } from "../lib/wm/model";

const model = WorldModel.fromJSON(JSON.parse(readFileSync(process.argv[2] ?? "results/results-model.json", "utf8")));
console.log(`mean Spearman(model return, true return) over 20 states: ${rankingCorrelation(model).toFixed(3)}`);
