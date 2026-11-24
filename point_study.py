"""'Bring me that one': can vision-language models follow a pointing gesture?

A person in a top-down room points at one of several labelled objects. Ground truth comes from geometry
(the object whose direction from the body is closest to the arm's direction). We vary how close the nearest
rival object is in angle, how far away the target is, and how many objects there are, and compare:
  image      - the rendered scene only
  image_ray  - the same scene with the pointing ray drawn (the system extends the gesture for the model)
  text       - no image; body, hand and object coordinates given as numbers
Everything runs locally with Ollama (temperature 0, seed 42).

Commands:  selftest | render | run | results
"""
import base64, json, math, os, random, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES, IMG = (os.path.join(HERE, d) for d in ("data", "results", os.path.join("data", "img")))
SCENES, OUT = os.path.join(DATA, "scenes.jsonl"), os.path.join(DATA, "answers.jsonl")
OLLAMA = "http://localhost:11434/api/chat"
MODELS = ["gemma4:12b", "qwen3.5:9b", "gemma3:12b"]
CONDS = ["image", "image_ray", "text"]
SEPS = [8, 15, 30]            # angle (deg) between the target and the closest rival, seen from the body
DISTS = {"near": 2.5, "far": 5.5}
NOBJ = [3, 6]
SEEDS = 8
ROOM, ARM = 10.0, 1.3
LETTERS = "ABCDEF"
COLORS = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]


def ang(v):
    return math.degrees(math.atan2(v[1], v[0]))


def adiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def inside(p, m=0.6):
    return m <= p[0] <= ROOM - m and m <= p[1] <= ROOM - m


def make_scene(sep, dist, n, seed):
    rng = random.Random(f"{sep}-{dist}-{n}-{seed}")
    while True:
        body = (rng.uniform(2.0, 8.0), rng.uniform(0.9, 2.0))
        th = rng.uniform(40, 140)
        d = DISTS[dist]
        u = (math.cos(math.radians(th)), math.sin(math.radians(th)))
        tgt = (body[0] + d * u[0], body[1] + d * u[1])
        side = rng.choice([-1, 1])
        rth = math.radians(th + side * sep)
        rd = d + rng.choice([-1, 1]) * rng.uniform(0.4, 0.8)   # offset in depth so small angles never overlap
        rival = (body[0] + rd * math.cos(rth), body[1] + rd * math.sin(rth))
        objs = [tgt, rival]
        tries = 0
        while len(objs) < n and tries < 500:
            tries += 1
            p = (rng.uniform(0.7, ROOM - 0.7), rng.uniform(0.7, ROOM - 0.7))
            if math.dist(p, body) < 1.3 or any(math.dist(p, q) < 1.0 for q in objs):
                continue
            if adiff(ang((p[0] - body[0], p[1] - body[1])), th) < sep + 5:
                continue
            objs.append(p)
        if len(objs) < n or not all(inside(p) for p in objs) or math.dist(tgt, rival) < 0.75:
            continue
        hand = (body[0] + ARM * u[0], body[1] + ARM * u[1])
        order = list(range(n))
        rng.shuffle(order)
        labelled = [{"label": LETTERS[i], "pos": [round(objs[k][0], 2), round(objs[k][1], 2)],
                     "color": COLORS[i]} for i, k in enumerate(order)]
        target = LETTERS[order.index(0)]
        rival_l = LETTERS[order.index(1)]
        return {"id": f"s{sep}_{dist}_n{n}_{seed}", "sep": sep, "dist": dist, "n": n, "seed": seed,
                "body": [round(body[0], 2), round(body[1], 2)], "hand": [round(hand[0], 2), round(hand[1], 2)],
                "theta": round(th, 1), "objects": labelled, "target": target, "rival": rival_l}


def truth(sc):
    """Geometric ground truth: the object closest in angle to the body->hand direction."""
    th = ang((sc["hand"][0] - sc["body"][0], sc["hand"][1] - sc["body"][1]))
    return min(sc["objects"], key=lambda o: adiff(ang((o["pos"][0] - sc["body"][0], o["pos"][1] - sc["body"][1])), th))["label"]


def heuristics(sc):
    """What simple shortcuts would answer: object nearest the hand, nearest the body."""
    near_hand = min(sc["objects"], key=lambda o: math.dist(o["pos"], sc["hand"]))["label"]
    near_body = min(sc["objects"], key=lambda o: math.dist(o["pos"], sc["body"]))["label"]
    return near_hand, near_body


def all_scenes():
    return [make_scene(s, d, n, k) for s in SEPS for d in DISTS for n in NOBJ for k in range(SEEDS)]


# ---------- rendering ----------
def render(sc, ray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, ROOM); ax.set_ylim(0, ROOM); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), ROOM, ROOM, color="#f3f1ea"))
    ax.add_patch(plt.Rectangle((0.05, 0.05), ROOM - 0.1, ROOM - 0.1, fill=False, lw=3, ec="#555"))
    b, h = sc["body"], sc["hand"]
    if ray:
        u = ((h[0] - b[0]) / ARM, (h[1] - b[1]) / ARM)
        end = (h[0] + 15 * u[0], h[1] + 15 * u[1])
        ax.plot([h[0], end[0]], [h[1], end[1]], ls="--", lw=1.6, color="#1f77b4", zorder=1)
    ax.add_patch(plt.Circle(b, 0.35, color="#1f77b4", zorder=3))
    ax.plot([b[0], h[0]], [b[1], h[1]], lw=7, color="#1f77b4", solid_capstyle="round", zorder=2)
    ax.add_patch(plt.Circle(h, 0.13, color="#0b3c5d", zorder=4))
    ax.text(b[0], b[1] - 0.75, "person", ha="center", va="top", fontsize=11, color="#0b3c5d")
    for o in sc["objects"]:
        ax.add_patch(plt.Circle(o["pos"], 0.32, color=o["color"], zorder=3))
        ax.text(o["pos"][0] + 0.45, o["pos"][1] + 0.35, o["label"], fontsize=15, weight="bold", zorder=5)
    path = os.path.join(IMG, f"{sc['id']}{'_ray' if ray else ''}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ---------- prompts ----------
SYS = "You answer questions about scenes. Reply only with JSON."
Q_IMG = ("This is a top-down view of a room. The person (blue circle) is pointing with their arm (the thick blue "
         "line ending in a dark hand) at one of the lettered objects.{ray} Which object is the person pointing at? "
         'Reply in JSON: {{"object": "<letter>"}}')
RAY_TXT = " The dashed line continues the direction of the arm."


def q_text(sc):
    objs = "; ".join(f"{o['label']} at ({o['pos'][0]}, {o['pos'][1]})" for o in sc["objects"])
    return ("A room seen from above, coordinates in metres (x to the right, y up). A person stands with their body "
            f"at ({sc['body'][0]}, {sc['body'][1]}) and points with their arm; their hand is at "
            f"({sc['hand'][0]}, {sc['hand'][1]}), so they point from the body through the hand. Objects: {objs}. "
            'Which object is the person pointing at? Reply in JSON: {"object": "<letter>"}')


def ask(model, prompt, image=None):
    msg = {"role": "user", "content": prompt}
    if image:
        with open(image, "rb") as f:
            msg["images"] = [base64.b64encode(f.read()).decode()]
    extra = {"think": False} if model.startswith(("qwen3", "gemma4")) else {}
    body = json.dumps({"model": model, "stream": False, "format": "json", **extra,
                       "messages": [{"role": "system", "content": SYS}, msg],
                       "options": {"temperature": 0, "seed": 42, "num_predict": 200, "num_ctx": 4096}}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:
            print("retry", attempt + 1, e, flush=True)
            time.sleep(15)
    raise RuntimeError("Ollama failed 3 times")


def parse(raw, n):
    try:
        v = str(json.loads(raw).get("object", "")).strip().upper()[:1]
    except Exception:
        return None
    return v if v in LETTERS[:n] else None


# ---------- commands ----------
def cmd_selftest():
    sc = all_scenes()
    assert len(sc) == len(SEPS) * len(DISTS) * len(NOBJ) * SEEDS
    for s in sc:
        assert truth(s) == s["target"], s["id"]
        th = s["theta"]
        b = s["body"]
        devs = sorted(adiff(ang((o["pos"][0] - b[0], o["pos"][1] - b[1])), th) for o in s["objects"])
        assert devs[0] < 2 and abs(devs[1] - s["sep"]) < 2, (s["id"], devs[:2])
    ids = [s["id"] for s in sc]
    assert len(set(ids)) == len(ids)
    for s in sc:
        assert "target" not in q_text(s).lower()   # the prompt never names the answer
    print(f"selftest ok: {len(sc)} scenes, truth = intended target in all, rival at the intended angle")


def cmd_render():
    os.makedirs(IMG, exist_ok=True)
    sc = all_scenes()
    with open(SCENES, "w") as f:
        for s in sc:
            render(s, False); render(s, True)
            f.write(json.dumps(s) + "\n")
    print(f"rendered {len(sc)} scenes x 2 images -> {IMG}")


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def cmd_run():
    sc = load(SCENES)
    done = {(r["model"], r["cond"], r["id"]) for r in load(OUT)}
    todo = [(m, c, s) for m in MODELS for c in CONDS for s in sc if (m, c, s["id"]) not in done]
    print(f"{len(todo)} calls to make", flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        for i, (m, c, s) in enumerate(todo):
            if c == "text":
                raw = ask(m, q_text(s))
            else:
                ray = c == "image_ray"
                raw = ask(m, Q_IMG.format(ray=RAY_TXT if ray else ""),
                          os.path.join(IMG, f"{s['id']}{'_ray' if ray else ''}.png"))
            ans = parse(raw, s["n"])
            nh, nb = heuristics(s)
            f.write(json.dumps({"model": m, "cond": c, "id": s["id"], "sep": s["sep"], "dist": s["dist"], "n": s["n"],
                                "answer": ans, "target": s["target"], "rival": s["rival"], "near_hand": nh,
                                "near_body": nb, "raw": raw[:300]}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)} {m} {c} {s['id']} -> {ans} (target {s['target']})", flush=True)
    print("done", flush=True)


def cmd_results():
    import numpy as np, pandas as pd
    df = pd.DataFrame(load(OUT))
    os.makedirs(RES, exist_ok=True)
    df["correct"] = df.answer == df.target
    df["invalid"] = df.answer.isna()
    df["picked_rival"] = df.answer == df.rival
    wrong = df[~df.correct & ~df.invalid]
    rng = np.random.default_rng(0)

    def ci(x):
        x = np.asarray(x, float)
        bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        return f"{x.mean():.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"

    out = ["# 'Bring me that one': results", "",
           f"Answers: {len(df)} ({df.model.nunique()} models x {df.cond.nunique()} conditions x {df.id.nunique()} scenes). "
           f"Invalid: {int(df.invalid.sum())}. Chance: 0.33 with 3 objects, 0.17 with 6.", "",
           "## Accuracy by model and condition (95% bootstrap CI)", "", "| model | " + " | ".join(CONDS) + " |",
           "|---|" + "---|" * len(CONDS)]
    for m in MODELS:
        out.append(f"| {m} | " + " | ".join(ci(df[(df.model == m) & (df.cond == c)].correct) for c in CONDS) + " |")
    for col, title in [("sep", "angle to the closest rival (deg)"), ("dist", "target distance"), ("n", "number of objects")]:
        t = df.pivot_table(index=["model", "cond"], columns=col, values="correct", aggfunc="mean").round(2)
        out += ["", f"## Accuracy by {title}", "", "```", t.to_string(), "```"]
    df["target_is_nearest_hand"] = df.near_hand == df.target
    t = df.pivot_table(index=["model", "cond"], columns="target_is_nearest_hand", values="correct", aggfunc="mean").round(2)
    t.columns = ["target NOT nearest the hand", "target IS nearest the hand"]
    out += ["", "## Direction or proximity? Accuracy split by whether the target is also the object nearest the hand",
            f"(shortcut 'pick the object nearest the hand' alone scores {(df[df.cond == 'image'].target_is_nearest_hand).mean():.2f})",
            "", "```", t.to_string(), "```"]
    if len(wrong):
        e = wrong.groupby(["model", "cond"]).agg(wrong=("correct", "size"), rival=("picked_rival", "mean"),
                                                   nearest_hand=("near_hand", lambda s: (s == wrong.loc[s.index, "answer"]).mean()),
                                                   nearest_body=("near_body", lambda s: (s == wrong.loc[s.index, "answer"]).mean())).round(2)
        out += ["", "## When wrong, what did the model pick? (share of wrong answers)",
                "`rival` = the object at the next-closest angle; `nearest_hand` / `nearest_body` = the object closest "
                "to the hand / body (shortcuts that ignore direction).", "", "```", e.to_string(), "```"]
    # paired test: image vs image_ray, per model, sign-flip permutation on scene-level differences
    out += ["", "## Does drawing the ray help? (paired permutation test, Holm-corrected)", ""]
    ps = []
    for m in MODELS:
        a = df[(df.model == m) & (df.cond == "image")].set_index("id").correct.astype(float)
        b = df[(df.model == m) & (df.cond == "image_ray")].set_index("id").correct.astype(float)
        d = (b - a.reindex(b.index)).dropna().values
        obs = d.mean()
        perm = [abs((d * rng.choice([-1, 1], len(d))).mean()) >= abs(obs) for _ in range(10000)]
        ps.append([m, obs, (sum(perm) + 1) / 10001])
    order = sorted(range(len(ps)), key=lambda i: ps[i][2])
    running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1, ps[i][2] * (len(ps) - rank)))
        ps[i].append(running)
    for m, obs, p, ph in ps:
        out.append(f"- {m}: change {obs:+.2f}, p = {p:.4f}, Holm p = {ph:.4f}")
    open(os.path.join(RES, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    # figure: accuracy vs separation, per model, image vs ray vs text
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(MODELS), figsize=(12, 3.6), sharey=True)
    for ax, m in zip(axs, MODELS):
        for c, st in zip(CONDS, ["-o", "-s", "--^"]):
            g = df[(df.model == m) & (df.cond == c)].groupby("sep").correct.mean()
            ax.plot(g.index, g.values, st, label=c)
        ax.set_title(m); ax.set_xlabel("angle to closest rival (deg)"); ax.set_xticks(SEPS); ax.set_ylim(0, 1.05)
        ax.grid(alpha=.3)
    axs[0].set_ylabel("accuracy"); axs[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_accuracy.png"), dpi=130)
    print("\n".join(out))


if __name__ == "__main__":
    {"selftest": cmd_selftest, "render": cmd_render, "run": cmd_run, "results": cmd_results}[sys.argv[1]]()
