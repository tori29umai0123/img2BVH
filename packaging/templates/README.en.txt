image2BVH portable
==================

Standalone WebUI: extract people from a single image, estimate full-body
pose, and export per-person BVH motion files. Runs on Windows 10 / 11
(x64).

Japanese: README.txt


Usage
-----

For detailed usage instructions and examples, see the following article
(in Japanese; machine-translate as needed - the screenshots and UI flow
are self-explanatory).

    https://note.com/tori29umai/n/n66e124812f69


Install
-------

1. Copy image2bvh-__VERSION__-portable.exe to the folder where you want
   the app to live (e.g. C:\Users\<YourName>\Apps\ or any writable
   folder).

2. Double-click the .exe and click "Yes" on the confirmation dialog.

3. An image2bvh\ folder is created next to the .exe and ~9 GB of files
   are extracted (takes a few minutes).

The extraction destination is fixed to the directory containing the
.exe. To install elsewhere, move the .exe to the desired folder first,
then run it.


Run
---

Double-click image2bvh\image2bvh.exe in the extracted folder.

The WebUI opens in your default browser at http://127.0.0.1:7860 after a
few tens of seconds. First launch takes an extra 30 sec - 1 min for the
MHR rest skeleton bake and Triton JIT compile; subsequent launches reuse
the cache.


Requirements
------------

OS:       Windows 10 / 11 x64

GPU:      NVIDIA RTX recommended (driver R580 or newer)
          - CUDA Toolkit install is NOT required (the bundled torch
            wheel ships the CUDA 13.0 runtime DLLs).
          - If no compatible GPU is present, the app falls back to CPU
            inference automatically (several times slower).

Disk:     ~10 GB free after extraction


Uninstall
---------

Delete the image2bvh\ folder.

This app never writes to the Windows registry. All state - config
(config.ini), model caches, temporary BVH outputs (tmp\), Triton JIT
cache (triton-cache\) - lives inside the extracted image2bvh\ folder.
Delete the folder and no trace remains on your machine.


Troubleshooting
---------------

* First launch is unexpectedly slow
    MHR rest-skeleton bake (one-time, cached to runtime\mhr_rest.json).
    Usually completes in 30 sec - 1 min.

* "GPU not recognised" / "CPU-only mode"
    Confirm NVIDIA driver is R580 or newer (check with nvidia-smi).

* Out of memory during GPU inference
    Set the environment variable CUDA_VISIBLE_DEVICES= (empty) before
    launching to force CPU mode.

* Browser doesn't open automatically
    Open http://127.0.0.1:7860 manually.

* Windows Defender / SmartScreen warning
    The bundled EXE is unsigned (single-file PyInstaller build), which
    can trigger false positives. After verifying the source, click
    "More info" then "Run anyway".


License
-------

This app redistributes Meta's SAM 3 / SAM 3D Body / DINOv3 models. By
using it you agree to the following licenses.

  * SAM License (Meta SAM 3 / SAM 3D Body)
  * DINOv3 License (Meta DINOv3)
  * image2BVH's own MIT License

Full text of all three licenses ships in LICENSE_BUNDLE.txt.

Military, weapons, nuclear, intelligence, and ITAR-controlled uses are
prohibited (SAM / DINOv3 License §1.b.v). Review the full license text
before commercial use or large-scale redistribution.
