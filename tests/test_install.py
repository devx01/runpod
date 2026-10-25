"""install.sh against a local git repo in a temporary HOME (--skip-pip --no-start): clone, then pull a new commit;
the keys from stdin go to .env (mode 600) and never into the output. python -m pytest -q (from runpod/)."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

INSTALL = Path(__file__).resolve().parent.parent / "install.sh"
KEYS = "TUNNEL_KEY=" + "t" * 64 + "\nTRAINER_KEY=" + "r" * 64 + "\nHF_TOKEN=hf_" + "h" * 34 + "\n"


def git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True)


def repo(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(INSTALL, src / "install.sh")
    (src / "server.py").write_text("# v1\n")
    git("init", "-q", "-b", "main", cwd=src)
    git("add", "-A", cwd=src)
    git("commit", "-q", "-m", "first", cwd=src)
    return src


def run(tmp_path, src, keys="", *flags):
    env = {**os.environ, "HOME": str(tmp_path / "home"), "RUNPOD_POD_ID": "pod123"}
    (tmp_path / "home").mkdir(exist_ok=True)
    r = subprocess.run(["bash", str(INSTALL), "--repo", str(src), "--skip-pip", "--no-start", *flags],
                       input=keys, capture_output=True, text=True, env=env, timeout=60)
    return r.returncode, r.stdout + r.stderr


def test_clone_keys_then_pull(tmp_path):
    src = repo(tmp_path)
    code, out = run(tmp_path, src, KEYS)
    assert code == 0, out
    steps = [ln.split()[1] for ln in out.splitlines() if ln.startswith("::step")]
    assert steps == ["code", "keys", "detector", "python"]
    assert "::info commit=" in out and "first" in out and "3 key(s) written" in out
    assert "TUNNEL_KEY: 64 characters" in out and "HF_TOKEN: 37 characters" in out
    assert "t" * 64 not in out and "r" * 64 not in out and "h" * 34 not in out  # lengths only, never the values
    assert "::warn no shared/detector.py" in out
    assert out.strip().splitlines()[-1] == "::done pod=pod123 tunnel=wss://pod123-8765.proxy.runpod.net/tunnel"
    env_file = tmp_path / "home" / "runpod" / ".env"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert env_file.read_text().splitlines() == KEYS.strip().splitlines()

    (src / "server.py").write_text("# v2\n")  # a new commit on GitHub: the next install pulls it
    git("commit", "-q", "-am", "second", cwd=src)
    code, out = run(tmp_path, src, "HF_TOKEN=hf_new\n")
    assert code == 0 and "second" in out, out
    assert (tmp_path / "home" / "runpod" / "server.py").read_text() == "# v2\n"
    kept = dict(ln.split("=", 1) for ln in env_file.read_text().splitlines())
    assert kept == {"TUNNEL_KEY": "t" * 64, "TRAINER_KEY": "r" * 64, "HF_TOKEN": "hf_new"}  # only the sent key changed


def test_no_tunnel_key_is_refused(tmp_path):
    code, out = run(tmp_path, repo(tmp_path), "HF_TOKEN=hf_x\n")
    assert code == 1 and "::fail keys: no TUNNEL_KEY" in out


def test_a_bad_repo_fails_at_the_code_step(tmp_path):
    code, out = run(tmp_path, tmp_path / "nothing-here", KEYS)
    assert code == 1 and "::fail code: git clone" in out
