"""
Start the showcase. One command, checks first, then plays.

    py run.py                 loopback + LAN, port 4661
    py run.py --port 8080
    py run.py --local-only    do not expose it on the network
    py run.py --check         preflight only, start nothing

Everything it needs is checked before the brain is loaded, because a missing
file should be a one-line message rather than a stack trace forty seconds in.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
BUILD = ROOT / "build"
DATA = ROOT / "data"


def lan_ip():
    """The address other machines can reach. No traffic is actually sent."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))       # picks the default route's interface
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()


def preflight():
    """Everything that has to be true before the brain is worth loading."""
    ok = True

    graph = BUILD / "graph.npz"
    if graph.exists():
        print(f"  [ok]   brain            {graph.stat().st_size / 1e6:.0f} MB")
    else:
        ok = False
        print("  [MISSING] build/graph.npz - the brain itself.")
        print("            Build it once:  py build_graph.py")
        print("            (needs the connectome in data/; see GUITAR.md)")

    ann = DATA / "body-annotations.feather"
    if ann.exists():
        print(f"  [ok]   annotations      {ann.stat().st_size / 1e6:.0f} MB")
    else:
        ok = False
        print("  [MISSING] data/body-annotations.feather")
        print("            Soma sides drive the fretting readout and soma")
        print("            coordinates draw the neuron panel. Without it the")
        print("            fly cannot play. See GUITAR.md for the download.")

    missing = []
    for mod in ("numpy", "scipy", "pandas", "pyarrow", "fastapi", "uvicorn"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        ok = False
        print(f"  [MISSING] python packages: {', '.join(missing)}")
        print("            py -m pip install -r requirements-guitar.txt")
    else:
        print("  [ok]   packages")

    pieces = sorted(BUILD.glob("set_piece*.json"))
    if pieces:
        total = 0
        for p in pieces:
            try:
                total += len(json.loads(p.read_text()).get("notes") or [])
            except Exception:
                pass
        live = [p for p in pieces if "live" in p.name]
        print(f"  [ok]   set pieces       {len(pieces)} "
              f"({total} notes total)"
              + (f", including {len(live)} composed live" if live else ""))
    else:
        print("  [warn] no set pieces in build/. It will fall back to live")
        print("         notes, which sound repetitive - live composition is")
        print("         about 60x slower than playback. Compose one with:")
        print("         py compose.py --seconds 30 --out build/set_piece_a.json")

    probe = BUILD / "guitar_probe.json"
    print(f"  [{'ok' if probe.exists() else 'warn'}]   circuit probe    "
          + ("measured" if probe.exists()
             else "absent; using the default strike threshold"))
    return ok


def port_free(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=4661)
    ap.add_argument("--local-only", action="store_true",
                    help="bind loopback only; nobody else can reach it")
    ap.add_argument("--check", action="store_true", help="preflight, then stop")
    ap.add_argument("--loop", type=int, default=None,
                    help="corpus loop index for the LIVE fly (not the pieces)")
    args = ap.parse_args()

    # Line-buffered, or these prints sit in the buffer until the process ends
    # - and this process hands off to the server, so that is never. Redirected
    # to a log, the URLs would simply never appear.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("\nthe fly plays electric guitar\n")
    print("preflight")
    ok = preflight()
    if not ok:
        print("\nnot starting - fix the above first.\n")
        return 1
    if args.check:
        print("\npreflight only; nothing started.\n")
        return 0

    if not port_free(args.port):
        print(f"\n  port {args.port} is already in use - something is running.")
        print(f"  Either use it, or stop that first, or pass --port <other>.\n")
        return 1

    host = "127.0.0.1" if args.local_only else "0.0.0.0"
    ip = None if args.local_only else lan_ip()

    print("\nopen")
    print(f"  you          http://127.0.0.1:{args.port}/")
    if ip:
        print(f"  same wifi    http://{ip}:{args.port}/")
    else:
        print("  same wifi    (loopback only)")

    print("\nnotes for showing it")
    print("  * click the page once - browsers block audio until you do")
    print("  * it keeps what it composes; leave it running and by showtime")
    print("    it is performing music it wrote in the room")
    print("  * venue wifi often blocks device-to-device. Test the second URL")
    print("    from a phone before you need it; a hotspot is the fallback")
    print("\nCtrl+C to stop\n")

    env = dict(os.environ, FLY_GUITAR_HOST=host,
               FLY_GUITAR_PORT=str(args.port))
    cmd = [sys.executable, str(ROOT / "guitarist.py"), "--port", str(args.port)]
    if args.loop is not None:
        cmd += ["--loop", str(args.loop)]
    try:
        return subprocess.call(cmd, env=env, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\nstopped.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
