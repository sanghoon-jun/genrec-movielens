"""Run genrec.py on a Modal GPU; runs/, data/ and the HF cache persist on the "genrec" volume.

GPU=H100 uvx modal run --detach scripts/modal_run.py --name run30 --args "--backbone Qwen/Qwen2.5-7B --prompt profile ..."
uvx modal volume get genrec runs/<run_dir> runs/    # fetch results
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).parent.parent
GPU = os.environ.get("GPU", "A100-80GB")  # e.g. GPU=H100

app = modal.App("genrec")
vol = modal.Volume.from_name("genrec", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(REPO))  # same locked versions as local runs
    .env({"HF_HOME": "/vol/hf"})
    .add_local_file(REPO / "common.py", "/root/app/common.py")
    .add_local_file(REPO / "genrec.py", "/root/app/genrec.py")
)


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, volumes={"/vol": vol}, timeout=24 * 3600)
def train(name: str, args: str):
    import os, shlex, subprocess, sys, time

    os.chdir("/root/app")
    for d in ("runs", "data"):  # common.py writes next to itself; point both at the volume
        os.makedirs(f"/vol/{d}", exist_ok=True)
        os.path.exists(d) or os.symlink(f"/vol/{d}", d)
    p = subprocess.Popen([sys.executable, "-u", "genrec.py", *shlex.split(args)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = time.time()
    with open(f"/vol/{name}.out", "w") as out:
        for line in p.stdout:
            print(line, end="")
            out.write(line)
            out.flush()
            if time.time() - last > 600:  # checkpoints and log survive a crash
                vol.commit()
                last = time.time()
        out.write(f"TRAINING_EXIT={p.wait()}\n")
    vol.commit()


@app.local_entrypoint()
def main(name: str, args: str):
    call = train.spawn(name, args)  # returns at once; with --detach the run does not depend on this client
    print(f"spawned {call.object_id}; log: uvx modal volume get genrec {name}.out")
