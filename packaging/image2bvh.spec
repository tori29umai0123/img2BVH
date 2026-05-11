# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for image2BVH (Windows desktop build).

Run from the project root with the venv activated:

    pyinstaller --noconfirm packaging\\image2bvh.spec

Output lands in ``dist\\image2bvh\\`` (the --onedir tree that Inno Setup
ingests in installer\\image2bvh.iss).

Design notes:
  * --onedir, not --onefile. cu130 nightly + triton + transformers is
    ~6 GB; onefile would re-extract that to %TEMP% on every launch and
    invalidate Triton's JIT cache.
  * console=False because Gradio opens a browser tab — no need for a
    persistent cmd window.
  * No UPX. UPX corrupts torch's .pyd files and re-flagging from AV
    engines goes through the roof for compressed PyInstaller bundles.
"""
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

# SPECPATH is the directory containing this .spec file (``packaging/``).
PROJECT_ROOT = Path(SPECPATH).parent

# ---------------------------------------------------------------------------
# Heavy packages — pull every .py + data file + .dll/.pyd via collect_all.
# Order doesn't matter; we just accumulate triples.
# ---------------------------------------------------------------------------
COLLECT_TARGETS = [
    # Torch stack — the wheels bundle CUDA/cuDNN/Triton DLLs inside torch/lib.
    # collect_all picks those up automatically. End-user PCs need a recent
    # NVIDIA driver only — no CUDA Toolkit install required.
    "torch",
    "torchvision",
    "triton",
    # WebUI + 3D preview frontends
    "gradio",
    "gradio_client",
    # gradio deps that ship a runtime version.txt (PyInstaller static
    # analysis grabs the .py but not the data file → FileNotFoundError
    # at import unless we collect_all them explicitly).
    "safehttpx",
    "groovy",
    "plotly",
    # SAM3 / SAM 3D Body loaders
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "accelerate",
    "timm",
    "einops",
    # MHR / mesh deps
    "smplx",
    "trimesh",
    "omegaconf",
    "yacs",
    "pytorch_lightning",
    "lightning_fabric",
    "lightning_utilities",
    "roma",
    "joblib",
    "braceexpand",
    "loguru",
    "termcolor",
    # Numerics & image IO
    "scipy",
    "PIL",
    "cv2",
]

datas: list = []
binaries: list = []
hiddenimports: list = []

for pkg in COLLECT_TARGETS:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# transformers does dynamic model class registration via importlib —
# belt-and-suspenders: also pull every ``transformers.models.*`` submodule.
hiddenimports += collect_submodules("transformers.models")

# Vendored DINOv3 / SAM 3D Body stack. The vendored package's pyproject.toml
# isn't visible to PyInstaller's auto-discovery because the package lives
# inside our source tree (``image2bvh/vendor/sam_3d_body/``) rather than
# site-packages, so we have to spell out the submodule sweep + data globs.
VENDOR_PKG = "image2bvh.vendor.sam_3d_body"
hiddenimports += collect_submodules(VENDOR_PKG)
datas += collect_data_files(
    VENDOR_PKG,
    includes=[
        "**/*.yaml", "**/*.yml",
        "**/*.json",
        "**/*.npy", "**/*.npz",
        "**/*.pkl",
        "**/*.txt", "**/*.md",
        # dinov3_repo is loaded via ``torch.hub.load(..., source="local")``
        # which does on-disk imports of hubconf.py and the dinov3/* tree —
        # PYZ-only inclusion isn't enough; the .py files have to exist as
        # real files on disk under _internal\image2bvh\vendor\....
        "**/dinov3_repo/**/*.py",
    ],
)

# importlib.metadata.version() lookups happen at import time for several deps.
# Without dist-info, those calls raise PackageNotFoundError and break startup.
for pkg in (
    "torch", "torchvision",
    "transformers", "tokenizers", "huggingface_hub", "safetensors",
    "accelerate", "timm",
    "gradio", "gradio_client",
    "pytorch_lightning", "lightning_fabric", "lightning_utilities",
    "smplx",
):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        # Package missing metadata is non-fatal — collect_all already
        # pulled the .py files; only the version() check would fail.
        pass

# ---------------------------------------------------------------------------
# Bundled model weights and license texts.
#   - SAM3:    runtime/models/sam3/   (skip duplicate sam3.pt, 3.3 GB)
#   - SAM3DB:  runtime/models/sam3dbody/
#   - License: LICENSES/ + per-model LICENSE files (required by
#              SAM/DINOv3 license §1.b.i for redistribution).
# ---------------------------------------------------------------------------
runtime_models = PROJECT_ROOT / "runtime" / "models"
if runtime_models.is_dir():
    for sub in runtime_models.rglob("*"):
        if not sub.is_file():
            continue
        if sub.name == "sam3.pt":
            # 3.3 GB Meta-format checkpoint; HF transformers loads from
            # model.safetensors instead. Saves 3.3 GB of installer size.
            continue
        rel_parent = sub.parent.relative_to(PROJECT_ROOT)
        datas.append((str(sub), str(rel_parent)))

for top in ("LICENSE", "NOTICE", "README.md", "config.ini"):
    p = PROJECT_ROOT / top
    if p.is_file():
        datas.append((str(p), "."))

licenses_dir = PROJECT_ROOT / "LICENSES"
if licenses_dir.is_dir():
    for lic in licenses_dir.iterdir():
        if lic.is_file():
            datas.append((str(lic), "LICENSES"))

# ---------------------------------------------------------------------------
# Hidden imports — places where PyInstaller's static analyser can't see
# through dynamic loading.
# ---------------------------------------------------------------------------
hiddenimports += [
    "image2bvh",
    "image2bvh.__main__",
    "image2bvh.gradio_app",
    "image2bvh.bootstrap",
    "image2bvh.bvh_export",
    "image2bvh.bvh_writer",
    "image2bvh.config",
    "image2bvh.hand_inference",
    "image2bvh.paths",
    "image2bvh.pose",
    "image2bvh.segmentation",
    "image2bvh.vram",
    "image2bvh.vendor.sam_3d_body",
    # SAM3 transformers entry points (transformers' Auto* classes look these
    # up via importlib at call time).
    "transformers.models.sam3",
    "transformers.models.sam3.modeling_sam3",
    "transformers.models.sam3.processing_sam3",
    "transformers.models.sam3.image_processing_sam3",
    "transformers.models.sam3.configuration_sam3",
    # pytorch-lightning's rank-zero util is imported by sam_3d_body's
    # base_lightning_module.
    "pytorch_lightning.utilities.rank_zero",
]

# ---------------------------------------------------------------------------
# Excludes — heavyweight packages that nothing on the inference path
# imports. Excluding them shrinks the installer without breaking anything.
# Mirrors the "deliberately excluded" list from pyproject.toml.
# ---------------------------------------------------------------------------
excludes = [
    "tkinter",
    "matplotlib",
    "IPython",
    "jupyter",
    "jupyter_client",
    "jupyter_core",
    "notebook",
    "pytest",
    "sphinx",
    # Listed as deliberately excluded from the inference path in pyproject.toml.
    "detectron2",
    "xformers",
    "pyrender",
    "ftfy",
]

# ---------------------------------------------------------------------------
# Analysis / build steps. (PyInstaller 6.x signature; ``cipher`` removed.)
# ---------------------------------------------------------------------------
a = Analysis(
    [str(PROJECT_ROOT / "image2bvh" / "__main__.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT_ROOT / "packaging" / "pyinstaller_runtime_hook.py")],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="image2bvh",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "packaging" / "image2bvh.ico") if (PROJECT_ROOT / "packaging" / "image2bvh.ico").is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="image2bvh",
)
