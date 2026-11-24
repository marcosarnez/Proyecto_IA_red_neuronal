"""Rutas Tcl/Tk locales para el Python 3.12 utilizado en este equipo."""
import os
import sys
import ctypes
from pathlib import Path


def configurar_tk():
    library = Path(__file__).resolve().parent / ".runtime" / "tcl"
    if os.name != "nt" or not (library / "tcl8.6" / "init.tcl").exists():
        return
    # No aplicar scripts 8.6.12 a otra instalacion cuyo DLL use otra version.
    try:
        runtime = ctypes.WinDLL(str(Path(sys.base_prefix) / "DLLs" / "tcl86t.dll"))
        values = [ctypes.c_int() for _ in range(4)]
        runtime.Tcl_GetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4
        runtime.Tcl_GetVersion.restype = None
        runtime.Tcl_GetVersion(*(ctypes.byref(value) for value in values))
        compatible = tuple(value.value for value in values[:3]) == (8, 6, 12)
    except OSError:
        compatible = False
    if compatible:
        os.environ["TCL_LIBRARY"] = str(library / "tcl8.6")
        os.environ["TK_LIBRARY"] = str(library / "tk8.6")
