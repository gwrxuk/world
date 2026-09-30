// Server-side training: the same TypeScript model code runs in a Node.js
// route handler. POST {"steps": 500, "episodes": 100} returns the trained
// weights and the loss curve, which the browser can load into its worker.

import { NextResponse } from "next/server";
import { collectEpisode } from "@/lib/wm/env";
import { DEFAULT_CONFIG, WorldModel } from "@/lib/wm/model";
import { makeRng } from "@/lib/wm/nn";

export const runtime = "nodejs";
export const maxDuration = 300;

const MAX_STEPS = 2000;
const MAX_EPISODES = 400;

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as { steps?: number; episodes?: number; seed?: number };
  const steps = Math.min(MAX_STEPS, Math.max(1, Math.floor(body.steps ?? 500)));
  const nEpisodes = Math.min(MAX_EPISODES, Math.max(1, Math.floor(body.episodes ?? 100)));
  const seed = Math.floor(body.seed ?? 42);

  const rng = makeRng(seed);
  const episodes = Array.from({ length: nEpisodes }, () => collectEpisode(rng));
  const model = new WorldModel({ ...DEFAULT_CONFIG, seed });
  const curve: { step: number; total: number }[] = [];
  const t0 = Date.now();
  for (let i = 1; i <= steps; i++) {
    const l = model.trainStep(episodes);
    if (i % 50 === 0 || i === steps) curve.push({ step: i, total: l.total });
  }
  return NextResponse.json({ model: model.toJSON(), curve, seconds: (Date.now() - t0) / 1000 });
}
