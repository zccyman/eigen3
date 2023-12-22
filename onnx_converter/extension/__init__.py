try:
    from .libs import pyops
except:
    import os
    import ctypes
    import platform
    import site
    import glob

    # Get the current Python version
    version = platform.python_version()
    # Split the version into major, minor, and patch parts
    major, minor, patch = version.split(".")
    # Concatenate the major and minor parts to get the major.minor version
    python_version = ".".join([major, minor])

    # Find the site packages directory that contains the given Python version
    python_path = None
    for path in site.getsitepackages():
        if f"python{python_version}" in path:
            python_path = os.path.join(path, "onnx_converter/extension")
            break

    # If a path was found, load the library from that path
    if python_path:
        # Get the current value of the LD_LIBRARY_PATH environment variable
        ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
        # Compute the path to the library we want to load
        new_library_path = os.path.join(python_path, "libs")
        # If the library path is not already in LD_LIBRARY_PATH, load the library and update LD_LIBRARY_PATH
        if new_library_path not in ld_library_path:
            for lib in glob.glob(os.path.join(new_library_path, "*.so.*")):
                ctypes.cdll.LoadLibrary(lib)
            ld_library_path = ':'.join([ld_library_path, new_library_path])
            os.environ['LD_LIBRARY_PATH'] = ld_library_path
                
    from .libs import pyops