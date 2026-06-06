#!/usr/bin/env python3
"""TalentLens - a config-driven, embeddings-based candidate ranker.

The role is described by FILES, not code: a `description.md` (the full JD, read as
intent and embedded) and a `spec.json` (skill families, weights, seniority band and
the credibility rules). Drop a new folder under `jobs/` and the same engine ranks
for a different role - nothing here is hardcoded to one job.

    fit = (w_sem·semantic + w_skill·skill_coverage + w_sen·seniority)
          × credibility       # multiplier 0.15–1.0: kills keyword-stuffers
          × availability       # multiplier 0.55–1.0: down-weights the unreachable

Semantics come from sentence-transformers (all-MiniLM-L6-v2) and are cached to disk,
so only the first run pays for encoding. CPU-only, offline after the first encode.

    python rank.py --job jobs/senior_ai_engineer --candidates candidates.jsonl
    python rank.py --sample
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- job

class Job:
    """Everything about a role, loaded from a folder (description.md + spec.json)."""

    def __init__(self, folder):
        self.folder = folder
        self.description = open(os.path.join(folder, "description.md"), encoding="utf-8").read()
        self.spec = json.load(open(os.path.join(folder, "spec.json"), encoding="utf-8"))
        self.must = self.spec["must_have"]
        self.nice = self.spec.get("nice_to_have", {})
        self.w = self.spec["weights"]
        self.sen = self.spec["seniority"]
        c = self.spec["credibility"]
        self._eng = self._rx(c["engineering_titles"])
        self._off = self._rx(c["off_domain_titles"])
        self._spec_off = self._rx(c.get("off_domain_specialist", []))
        self._svc = self._rx(c["services_companies"])
        self._fw = self._rx(c.get("framework_only", []))
        self.avail = self.spec.get("availability", {})

    @staticmethod
    def _rx(terms):
        return re.compile("|".join(re.escape(t.strip()) for t in terms), re.I) if terms else None


# --------------------------------------------------------------------- candidate

def load(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
            rows = data if isinstance(data, list) else [data]
        else:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows[:limit] if limit else rows


def document(c):
    """A rich, readable text for the candidate - what we embed and search."""
    p = c.get("profile", {}) or {}
    out = [p.get("headline", ""), p.get("summary", "")]
    if p.get("current_title"):
        out.append(f"Currently {p['current_title']} at {p.get('current_company','')} "
                   f"({p.get('current_industry','')}).")
    for r in (c.get("career_history") or [])[:6]:
        out.append(f"{r.get('title','')} at {r.get('company','')} "
                   f"[{r.get('industry','')}]: {r.get('description','')}")
    sk = [f"{s.get('name','')} ({s.get('proficiency','')}, {s.get('duration_months',0)}mo)"
          for s in (c.get("skills") or [])]
    if sk:
        out.append("Skills: " + ", ".join(sk))
    return "  ".join(b for b in out if b and b.strip())


def evidence(c, text_lc, families, limit=3):
    """For each family, the concrete skills/terms that prove it (for the reasoning)."""
    skills = [(s.get("name", ""), s.get("proficiency", "")) for s in (c.get("skills") or [])]
    hit = {}
    for fam, kws in families.items():
        found, seen = [], set()
        for name, prof in skills:                      # prefer naming real skills
            nl = name.lower()
            if any(k in nl for k in kws) and nl not in seen:
                seen.add(nl); found.append(name)
            if len(found) >= limit:
                break
        if not found:                                  # else fall back to JD keyword in prose
            for k in kws:
                if k in text_lc:
                    found.append(k); break
        if found:
            hit[fam] = found
    return hit


# ------------------------------------------------------------------- components

def seniority_fit(yoe, sen):
    a, b = sen["ideal"]
    c, d = sen["acceptable"]
    e, f = sen["soft"]
    if yoe <= 0:
        return 0.3
    if a <= yoe <= b:
        return 1.0
    if c <= yoe <= d:
        return 0.85
    if e <= yoe <= f:
        return 0.6
    return 0.35


def credibility(c, text_lc, job):
    """Multiplier in [0.15, 1.0] + human flags. Encodes the JD's explicit anti-patterns."""
    titles = [c.get("profile", {}).get("current_title", "")] + \
             [r.get("title", "") for r in (c.get("career_history") or [])]
    has_eng = any(job._eng and job._eng.search(t or "") for t in titles)
    cur = c.get("profile", {}).get("current_title", "") or ""
    yoe = float(c.get("profile", {}).get("years_of_experience") or 0)
    s, flags = 1.0, []

    if job._off and job._off.search(cur) and not has_eng:
        s *= 0.18; flags.append(f"current role '{cur}' is off-domain and no engineering/ML role appears in the career history")
    elif job._off and job._off.search(cur):
        s *= 0.72; flags.append(f"current title '{cur}' is not an engineering role")

    for sk in c.get("skills") or []:
        if sk.get("proficiency") in ("expert", "advanced") and (sk.get("duration_months") or 0) == 0:
            s *= 0.6; flags.append(f"claims {sk.get('proficiency')} in {sk.get('name')} with 0 months of use")
            break

    months = sum((r.get("duration_months") or 0) for r in (c.get("career_history") or []))
    if yoe and months > yoe * 12 * 1.6:
        s *= 0.72; flags.append("logged tenure exceeds stated years of experience")

    comps = [r.get("company", "") for r in (c.get("career_history") or [])]
    if comps and job._svc and all(job._svc.search(x or "") for x in comps):
        s *= 0.75; flags.append("entire career at IT-services firms (no product-company experience)")

    if job._spec_off and job._spec_off.search(text_lc) and "nlp" not in text_lc and "retrieval" not in text_lc:
        s *= 0.7; flags.append("CV/speech/robotics focus with little NLP/IR")

    if job._fw and job._fw.search(text_lc) and not any(k in text_lc for kws in job.must.values() for k in kws):
        s *= 0.7; flags.append("LLM exposure looks framework-only (LangChain) with no retrieval/ranking depth")

    return max(s, 0.15), flags


def availability_fit(c, av):
    sig = c.get("redrob_signals", {}) or {}
    a, notes = 1.0, []
    last = str(sig.get("last_active_date", ""))[:7]
    if last and last < av.get("active_recent_cutoff", "2026-03"):
        a *= 0.85
    if last and last < av.get("active_stale_cutoff", "2025-12"):
        a *= 0.85; notes.append(f"inactive since {last}")
    if (sig.get("recruiter_response_rate") or 0) < av.get("low_response_rate", 0.1):
        a *= 0.85; notes.append("very low recruiter-response rate")
    if sig.get("open_to_work_flag") is False:
        a *= 0.9
    return max(a, 0.55), notes


def coverage(text_lc, must, nice):
    fams = [f for f, kws in must.items() if any(k in text_lc for k in kws)]
    bonus = 0.12 * sum(1 for kws in nice.values() if any(k in text_lc for k in kws))
    return min(len(fams) / max(len(must), 1) + bonus, 1.0), fams


# --------------------------------------------------------------------- reasoning

def verdict(score):
    return ("Strong fit" if score >= 0.72 else "Good fit" if score >= 0.5
            else "Moderate fit" if score >= 0.3 else "Weak fit")


def reasoning(c, score, ev, fams_hit, nice_hit, yoe, n_must, cred_flags, avail_notes):
    p = c.get("profile", {}) or {}
    role = f"{p.get('current_title','?')} @ {p.get('current_company','?')} ({yoe:g}y)"
    parts = [f"{verdict(score)}. {role}."]

    if ev:
        cov = "; ".join(f"{fam} ({', '.join(items)})" for fam, items in ev.items())
        parts.append(f"Covers {len(fams_hit)}/{n_must} must-haves: {cov}.")
    else:
        parts.append(f"Covers 0/{n_must} must-haves with concrete evidence.")
    if nice_hit:
        parts.append("Plus " + ", ".join(nice_hit) + ".")

    sig = c.get("redrob_signals", {}) or {}
    resp = sig.get("recruiter_response_rate")
    act = str(sig.get("last_active_date", "?"))[:7]
    eng = f"Active {act}" + (f", {resp:.0%} recruiter response" if isinstance(resp, (int, float)) else "")
    parts.append(eng + (" (" + "; ".join(avail_notes) + ")" if avail_notes else "") + ".")

    if cred_flags:
        parts.append("Caveat: " + "; ".join(cred_flags[:2]) + ".")
    return " ".join(parts)


# -------------------------------------------------------------------- embeddings

def embed(docs, cache_key, rebuild=False):
    cdir = os.path.join(HERE, ".cache")
    os.makedirs(cdir, exist_ok=True)
    cf = os.path.join(cdir, f"emb_{cache_key}.npy")
    if os.path.exists(cf) and not rebuild:
        return np.load(cf)
    from sentence_transformers import SentenceTransformer
    vecs = SentenceTransformer(MODEL).encode(
        docs, batch_size=64, normalize_embeddings=True, show_progress_bar=True).astype("float32")
    np.save(cf, vecs)
    return vecs


def load_precomputed(emb_path, cands, docs):
    """Load candidate embeddings from an .npz (ids + vecs), aligned by candidate_id.

    Candidate embeddings are job-independent, so one precomputed file serves every
    role and lets the ranking step skip the (slow) encode entirely. Any candidate not
    found in the file is encoded on the fly, so partial files still work."""
    d = np.load(emb_path, allow_pickle=True)
    vecs = d["vecs"]                                  # load once (avoid re-decompressing)
    idx = {cid: k for k, cid in enumerate(d["ids"].tolist())}
    rows = np.array([idx.get(c["candidate_id"], -1) for c in cands])
    M = np.zeros((len(cands), vecs.shape[1]), dtype="float32")
    have = rows >= 0
    M[have] = vecs[rows[have]]
    missing = [i for i, ok in enumerate(have) if not ok]
    if missing:
        print(f"  {len(missing):,} candidates not in the precomputed file -> encoding those")
        from sentence_transformers import SentenceTransformer
        mv = SentenceTransformer(MODEL).encode([docs[i] for i in missing], batch_size=64,
                                               normalize_embeddings=True, show_progress_bar=True)
        for j, i in enumerate(missing):
            M[i] = mv[j]
    return M


def save_precomputed(emb_path, cands, M):
    ids = np.array([c["candidate_id"] for c in cands])
    np.savez(emb_path, ids=ids, vecs=M.astype("float32"))
    print(f"  saved precomputed embeddings -> {emb_path} ({len(ids):,} candidates)")


def jd_vector(text):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL).encode([text], normalize_embeddings=True)[0]


def unit(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="TalentLens config-driven ranker")
    ap.add_argument("--job", default=os.path.join(HERE, "jobs", "senior_ai_engineer"),
                    help="path to a job folder (description.md + spec.json)")
    ap.add_argument("--candidates", default=os.path.join(HERE, "sample_candidates.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "submission.csv"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--sample", action="store_true", help="rank the bundled 50-candidate sample")
    ap.add_argument("--rebuild", action="store_true", help="recompute the embedding cache")
    ap.add_argument("--emb", default="", help="precomputed embeddings .npz (ids+vecs); skips encoding")
    ap.add_argument("--save-emb", default="", help="after encoding, save embeddings to this .npz")
    args = ap.parse_args()
    if args.sample:
        args.candidates = os.path.join(HERE, "sample_candidates.json")

    job = Job(args.job)
    print(f"Job: {job.spec['title']} @ {job.spec.get('company','')}  (from {os.path.basename(args.job)})")

    t0 = time.time()
    cands = load(args.candidates, args.limit)
    docs = [document(c) for c in cands]
    print(f"Loaded {len(cands):,} candidates in {time.time()-t0:.1f}s")

    t = time.time()
    if args.emb:
        print(f"Using precomputed embeddings: {args.emb}")
        M = load_precomputed(args.emb, cands, docs)
    else:
        key = hashlib.md5(f"{os.path.abspath(args.candidates)}:{len(cands)}".encode()).hexdigest()[:10]
        M = embed(docs, key, args.rebuild)
    if args.save_emb:
        save_precomputed(args.save_emb, cands, M)
    sem = unit(M @ jd_vector(job.description))  # only the 1-sentence JD is encoded here
    print(f"Embeddings ready in {time.time()-t:.1f}s")

    n_must = len(job.must)
    rows = []
    fit = np.zeros(len(cands))
    for i, c in enumerate(cands):
        tlc = docs[i].lower()
        cov, fams = coverage(tlc, job.must, job.nice)
        yoe = float(c.get("profile", {}).get("years_of_experience") or 0)
        cred, cflags = credibility(c, tlc, job)
        av, anotes = availability_fit(c, job.avail)
        base = job.w["semantic"] * sem[i] + job.w["skills"] * cov + job.w["seniority"] * seniority_fit(yoe, job.sen)
        fit[i] = base * cred * av
        ev = evidence(c, tlc, job.must)
        nice_hit = [f for f, kws in job.nice.items() if any(k in tlc for k in kws)]
        rows.append(dict(c=c, sem=sem[i], ev=ev, fams=fams, nice=nice_hit, yoe=yoe,
                         cflags=cflags, anotes=anotes))

    fit = unit(fit)
    order = sorted(range(len(cands)), key=lambda i: (-fit[i], cands[i]["candidate_id"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, i in enumerate(order[:args.topk], 1):
            r = rows[i]
            why = reasoning(r["c"], fit[i], r["ev"], r["fams"], r["nice"], r["yoe"],
                            n_must, r["cflags"], r["anotes"])
            w.writerow([cands[i]["candidate_id"], rank, f"{fit[i]:.4f}", why])

    print("\nTop 8:")
    for i in order[:8]:
        r = rows[i]
        print(f"  {cands[i]['candidate_id']}  {fit[i]:.3f}  {verdict(fit[i]):12}  "
              f"{str(r['c'].get('profile',{}).get('current_title',''))[:30]:30}  {r['yoe']:g}y")
    print(f"\nWrote {args.out}  ({min(args.topk, len(order))} rows) in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
