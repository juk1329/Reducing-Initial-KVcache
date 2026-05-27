# SETUP — running Reducing-Initial-Cache on a new server

This repo is **only my method** (`Reducing-Initial-Cache`). It **reuses KVzip** at
runtime (imports its model wrapper, scoring, and eviction code). So on a new server
you must also clone KVzip. The other reference repos (SnapKV, streaming-llm,
duo-attention, KVCache-Factory, Long-Context-Data-Engineering) are **not imported**
by this code and are not required to run it.

> Using Claude Code on the new server? Hand it this file. The steps below are
> ordered and include every gotcha learned while setting up the source env.

---

## 0. Source environment (reference — what worked)

| | source server (`jk`) |
|---|---|
| OS / shell | Linux / bash |
| conda env | `jk`, **python 3.12.11** |
| GPU | NVIDIA RTX 3080 Ti, **compute capability 8.6 (sm_86)**, 12 GB |
| CUDA toolkit (nvcc) | 12.9 |
| torch | **2.8.0+cu128** (torch.version.cuda = 12.8) |
| transformers | **4.51.3**  *(hard requirement)* |
| flash-attn | **2.8.3** (built from source for sm_86) |
| KVzip kernel | `tiny_api_cuda` (built from `KVzip/csrc`, sm_86) |

Exact freezes for reference: `env/jk-conda-full.yml`, `env/jk-pip-freeze.txt`.
**Do not blindly copy these** to a different server — torch/flash-attn/the kernel
are compiled for this exact GPU+CUDA and must be rebuilt for the target machine.

### ⚠️ Gotchas (these bit us; don't repeat them)
1. **`transformers==4.51.3` is mandatory.** KVzip monkey-patches the attention
   `forward`; transformers ≥ 4.54 changed the signature (e.g. `past_key_value` →
   `past_key_values`) and silently breaks it. Newer transformers = wrong results / errors.
2. **Do NOT `pip install -e .` on KVzip.** Its `pyproject.toml` pins
   `torch==2.3.0` + `numpy==1.26`, which would downgrade/break the env. We only
   build its CUDA kernel and import the package via `sys.path` (handled automatically).
3. **flash-attn must match the server's GPU arch and torch.** Build it with
   `TORCH_CUDA_ARCH_LIST="<server sm>"`.
4. **The KVzip kernel must target the server's GPU arch.** Edit `KVzip/csrc/build.py`
   to add `arch=compute_XX,code=sm_XX` for the server GPU before building.

---

## 1. Directory layout

Clone this repo **and** KVzip as siblings:

```
<workspace>/
├── Reducing-Initial-KVcache/     # THIS repo
│   ├── ric/  run_niah.py  run_scbench.py  aggregate.py  args.py
│   ├── README.md  SETUP.md  environment.yml  requirements.txt  env/
└── KVzip/                         # git clone https://github.com/snu-mllab/KVzip
```

`ric/__init__.py` auto-resolves KVzip at `<parent-of-this-repo>/KVzip`. If you put
KVzip elsewhere, set the env var instead:
```bash
export KVZIP_DIR=/abs/path/to/KVzip
```

```bash
cd <workspace>
git clone git@github.com:juk1329/Reducing-Initial-KVcache.git
git clone https://github.com/snu-mllab/KVzip.git
```

---

## 2. Find the server's GPU compute capability (you need it for steps 4 & 5)

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# e.g. "NVIDIA A100 ..., 8.0"  -> sm_80   |   RTX 3080 Ti -> 8.6 -> sm_86
nvcc --version    # confirm a CUDA toolkit is installed (needed to build flash-attn + kernel)
```
Let `CC` = that number (e.g. `8.0`), and `SM` = it without the dot (e.g. `80`).

---

## 3. Create the conda env + portable deps

```bash
conda create -n jk python=3.12 -y
conda activate jk
pip install -r requirements.txt          # transformers==4.51.3 + eval/util deps (NO torch/flash-attn)
```

---

## 4. Install PyTorch for the server's CUDA

Pick the wheel matching the server's CUDA (check https://pytorch.org). Examples:
```bash
# CUDA 12.8 (what we used):
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# CUDA 12.1:
# pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```
Any recent torch 2.x compatible with transformers 4.51.3 is fine. Verify:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

---

## 5. Build flash-attn for the server GPU

```bash
TORCH_CUDA_ARCH_LIST="$CC" MAX_JOBS=8 pip install flash-attn --no-build-isolation
```
- Replace `$CC` (e.g. `"8.0"`). Restricting the arch keeps the build short.
- `MAX_JOBS` ~ number of CPU cores with ≥4 GB RAM each (we used 8 with 43 GB free).
- We got `flash-attn 2.8.3`. If the build fails on a very new/old torch, try a
  nearby flash-attn version (`pip install "flash-attn==2.8.3" --no-build-isolation`).

---

## 6. Build the KVzip CUDA kernel (`tiny_api_cuda`) for the server GPU

```bash
cd <workspace>/KVzip/csrc
```
Edit `build.py`: in the `cc_flag` block add your server's arch (keep existing lines).
We added these two lines for sm_86; change `86` to your `SM`:
```python
cc_flag.append("-gencode")
cc_flag.append("arch=compute_86,code=sm_86")   # <- set to your server: compute_<SM>,code=sm_<SM>
```
Then build + install the kernel (this does NOT install the KVzip package itself):
```bash
python build.py install
cd -
```

---

## 7. Verify the stack

```bash
python -c "import torch, transformers, flash_attn, tiny_api_cuda; \
print('torch', torch.__version__, '| tf', transformers.__version__, \
'| fa', flash_attn.__version__, '| kernel OK')"
```
Expect `transformers 4.51.3` and `kernel OK`.

---

## 8. Hugging Face access (gated models)

`llama3.2-*` are gated. Log in and accept the model license on HF first:
```bash
pip install -U huggingface_hub
huggingface-cli login        # paste a token with access to the model
```
Models download on first run. Open alternatives that need no gating: `qwen3-1.7b`,
`qwen3-4b` (KVzip model names; see `KVzip/model/load.py`).

---

## 9. Smoke test (small, fast) then full experiments

```bash
cd <workspace>/Reducing-Initial-KVcache
# tiny smoke test: confirms streaming prefill + answer run end to end
python run_niah.py -m llama3.2-1b --method ours --level per_token \
    --budget_max 2048 --budget_target 1024 --chunk 1024 --ctx_lens 2000,4000

# full experiments + comparison: see README.md §4
python aggregate.py        # build paper-ready tables from results/
```

If imports fail with "KVzip not found", set `KVZIP_DIR` (step 1).

---

## Quick checklist
- [ ] cloned this repo + KVzip as siblings (or set `KVZIP_DIR`)
- [ ] conda env `jk` (python 3.12) + `pip install -r requirements.txt`
- [ ] torch for the server's CUDA
- [ ] flash-attn built with `TORCH_CUDA_ARCH_LIST=<CC>`
- [ ] edited `KVzip/csrc/build.py` arch → built `tiny_api_cuda`
- [ ] `transformers==4.51.3` (verify!) and did NOT `pip install -e .` KVzip
- [ ] verify step 7 prints `kernel OK`
- [ ] HF login for gated models
