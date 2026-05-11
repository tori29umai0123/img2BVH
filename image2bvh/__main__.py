# Use an absolute import here. When invoked via ``python -m image2bvh``
# the package context exists and ``from .gradio_app import main`` would
# work fine — but PyInstaller freezes ``__main__.py`` as a top-level
# script (no parent package), so a relative import dies with
# ``ImportError: attempted relative import with no known parent package``.
# The absolute form works in both modes because ``image2bvh`` is on
# sys.path either way.
from image2bvh.gradio_app import main

if __name__ == "__main__":
    main()
