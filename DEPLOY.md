# Deploying the fly

## What the server actually needs

Two artefacts, 52 MB total, neither of them in git:

| file | size | what it is |
|---|---|---|
| `build/graph.npz` | 38 MB | the brain: 165,122 neurons, 10,228,000 signed synapses |
| `data/body-annotations.feather` | 14 MB | soma sides (the fretting readout is an L/R asymmetry) and soma coordinates (the neuron scatter) |

Optionally, and worth shipping because it is a result rather than an input:

| file | size | what it is |
|---|---|---|
| `build/set_piece.json` | ~100 KB | a composed set piece the page performs |
| `build/guitar_probe.json` | ~4 KB | the circuit measurement; sets the strike threshold |
| `build/guitar_gains.npz` | small | fitted per-cell-type gains, if you ran `train_guitar.py` |

**Build the brain once, locally, then ship the 38 MB.** The 508 MB connectome
download and the `pandas`/`pyarrow` build step do not need to happen on the
server, and on a small VM they are the difference between a working deploy and
an out-of-memory one.

## 1. Build the artefacts locally (once)

    py -m pip install -r requirements-guitar.txt

    mkdir -p data
    B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
    curl -o data/body-annotations.feather       "$B/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    curl -o data/body-neurotransmitters.feather "$B/body-neurotransmitters-male-cns-v1.0.feather"
    curl -o data/connectome-weights.feather     "$B/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"

    py build_graph.py          # expect 165,122 neurons / 10,228,000 edges
    py probe_guitar.py         # measures the circuit; writes guitar_probe.json
    py compose.py --seconds 45 # composes a set piece (~18 min)

`data/connectome-weights.feather` and `data/body-neurotransmitters.feather` are
inputs to `build_graph.py` only. Delete them afterwards; do **not** ship them.

## 2. Run it locally

    py guitarist.py
    # http://127.0.0.1:4661/

Binds loopback by default. Flags and env vars, nothing hardcoded:

    py guitarist.py --host 0.0.0.0 --port 8080 --loop 2 --sim-steps 100
    FLY_GUITAR_HOST=0.0.0.0 FLY_GUITAR_PORT=8080 py guitarist.py

## 3. Deploy to a VM

Any Linux box with **2 GB RAM and 1–2 vCPU**. This is CPU-bound: it simulates
165,122 neurons continuously and will happily use a whole core. There is no
GPU path worth taking — a batched-CUDA simulator speeds up many runs at once,
not a single one.

    # on the server
    git clone https://github.com/Palak11245/fly-guitar.git
    cd fly-guitar
    python3 -m venv .venv && . .venv/bin/activate
    pip install -r requirements-guitar.txt

    # from your machine, ship the two artefacts
    scp build/graph.npz               user@host:~/fly-guitar/build/
    scp build/set_piece.json          user@host:~/fly-guitar/build/
    scp build/guitar_probe.json       user@host:~/fly-guitar/build/
    scp data/body-annotations.feather user@host:~/fly-guitar/data/

`/etc/systemd/system/flyguitar.service`:

    [Unit]
    Description=The fly plays electric guitar
    After=network.target

    [Service]
    User=fly
    WorkingDirectory=/home/fly/fly-guitar
    Environment=FLY_GUITAR_HOST=127.0.0.1
    Environment=FLY_GUITAR_PORT=4661
    ExecStart=/home/fly/fly-guitar/.venv/bin/python guitarist.py
    Restart=always
    RestartSec=10

    [Install]
    WantedBy=multi-user.target

    sudo systemctl enable --now flyguitar
    journalctl -u flyguitar -f

Put nginx in front for TLS. The websocket needs the upgrade headers:

    location / {
        proxy_pass         http://127.0.0.1:4661;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 3600s;
    }

Without `proxy_read_timeout`, nginx closes the socket after 60 s and the page
falls back to reconnecting every minute. The page reconnects on its own, but
you will see it.

## 4. Docker

    FROM python:3.12-slim
    WORKDIR /app
    COPY requirements-guitar.txt .
    RUN pip install --no-cache-dir -r requirements-guitar.txt
    COPY . .
    ENV FLY_GUITAR_HOST=0.0.0.0 FLY_GUITAR_PORT=8080
    EXPOSE 8080
    CMD ["python", "guitarist.py"]

Mount the artefacts rather than baking 52 MB into the image:

    docker build -t fly-guitar .
    docker run -p 8080:8080 \
      -v "$PWD/build:/app/build" \
      -v "$PWD/data:/app/data" \
      fly-guitar

## 5. Platform notes

**The dividing line is persistent vs request-scoped**, not which vendor.
This process holds a 165,122-neuron brain in memory, steps it continuously,
and streams over a websocket that stays open. Anything that spins a container
up per request cannot host it.

- **Render — yes.** Web Services are persistent containers with native
  websocket support and `$PORT` injected (already handled). `render.yaml` in
  this repo is a working blueprint. Measured footprint is ~352 MB, so the
  512 MB plans fit with little headroom and the 2 GB plan is comfortable.
  Avoid the free tier: it spins down after 15 minutes idle, and a cold start
  both restarts the performance and — with no disk — discards everything the
  fly composed live.
- **Fly.io / Railway — yes**, same reasoning. Use a volume for the artefacts.
- **Vercel — no.** Not a tier problem, an architecture one. Vercel Functions
  are request-scoped, cap out at 60–300 s, and do not host a long-lived
  websocket server. The only thing it can usefully do is serve the static page
  against a backend hosted elsewhere, which is not worth splitting.
- **Lambda, Cloudflare Workers — no**, for the same reason.
- **Free tiers that sleep on idle will not work well.** The brain takes a
  couple of seconds to load and the simulation is the point; a cold start
  restarts the performance.
- **One process serves many viewers.** A watcher is a subscriber and never a
  controller, so the fly plays whether or not anyone is connected. Do not run
  multiple workers — you would get several unrelated flies and multiply the
  CPU cost. `uvicorn` is started single-process deliberately.

## 6. Checks after deploying

    curl -s localhost:4661/setup | head -c 300     # ready:true, neuron count
    curl -s localhost:4661/state | head -c 300     # a live tick
    curl -s localhost:4661/piece | head -c 200     # ready:true if a piece shipped

If `/setup` says `soma: null`, `body-annotations.feather` did not make it
across and the neuron scatter will be empty. If `/piece` says
`ready: false`, no set piece shipped and the page falls back to live notes —
which will sound repetitive, because live composition is ~60x slower than
playback. That is expected, and `compose.py` is the fix.

## 7. Security

What the exposed surface actually is, having gone looking for it.

**There is no user input.** The server parses nothing a client sends. The
websocket calls `receive_text()` only to notice a disconnect and discards the
result; there are no query parameters, no request bodies, no forms. A watcher
is a subscriber and never a controller, so there is no input to validate and
nothing a client can steer.

**Reviewed and clean**
- No `eval`, `exec`, `subprocess`, or `shell=True` anywhere in the guitar code.
- Both `np.load` calls pass `allow_pickle=False`. This matters: a pickled
  `.npz` is arbitrary code execution at load time, and `guitar_gains.npz` is a
  file an operator drops in.
- One `FileResponse`, a fixed path. No `StaticFiles`, no path traversal.
- CORS is off unless `FLY_GUITAR_ORIGINS` is set, so the API is not readable
  cross-origin by default. When it is set it allows `GET` only and sends no
  credentials.
- No credentials, tokens or keys in any of it, and no `.env`: the only
  settings are the `FLY_GUITAR_*` environment variables documented above.
- Server data interpolated into the page is escaped before it reaches
  `innerHTML`. `set_piece.json` is operator-supplied rather than user-supplied,
  but a section name must not be able to inject markup.
- Websocket fan-out is capped (`FLY_GUITAR_MAX_CLIENTS`, default 64) so a
  flood of connections cannot grow the client set without bound. The fly plays
  whether or not anyone is watching, so refusing an extra viewer costs nothing.

**Decide these before exposing it**
- **It binds loopback by default, and that is the safe default.** Setting
  `--host 0.0.0.0` publishes it. Put nginx in front and terminate TLS there.
- **There is no authentication, by design.** Anyone who can reach it can watch.
  That is the point of the page, but it does mean anyone who can reach it can
  also hold a websocket open. If it should not be public, put it behind basic
  auth in nginx or a private network.
- **The websocket does not check `Origin`.** Any page can open a socket and
  read the stream. The stream contains neuron rates and notes — nothing
  sensitive, and nothing a viewer can act on — but if you would rather only
  your own page connect, check `Origin` in the handler or have nginx do it.
- **It is CPU-bound and single-process.** That is its DoS profile: the
  simulation will use a core regardless of load, and many viewers cost little
  extra, but do not put it on a shared box you care about. Rate-limit at nginx.
- **Do not commit `build/graph.npz` or `data/`.** They are gitignored. The
  connectome is CC-BY and freely redistributable, but they are large and
  regenerable, and a git repo is not a CDN.

**Third-party**: the page loads three.js from cdnjs and fonts from Google
Fonts at runtime. If you would rather not depend on a CDN, or need a strict
CSP, vendor `three.min.js` into `web/` and serve it yourself — it is a single
608 KB file and the page needs no other JS.
