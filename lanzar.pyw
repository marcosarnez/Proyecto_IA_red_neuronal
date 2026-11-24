"""Inicio de escritorio con registro local de errores."""
from datetime import datetime
from pathlib import Path
import sys
import traceback


def iniciar():
    ruta_log = Path(__file__).resolve().parent / "salidas" / "registro_aplicacion.log"
    ruta_log.parent.mkdir(exist_ok=True)
    con_consola = sys.stderr is not None
    with ruta_log.open("a", encoding="utf-8", buffering=1) as log:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Inicio con {sys.executable}", file=log)
        if not con_consola:
            sys.stdout = sys.stderr = log
        try:
            # El script pedagógico configura Tcl/Tk al importarse.
            from Residuos_Medicos_Pedagogico import main

            if con_consola:
                print("Bibliotecas cargadas. Abriendo la aplicacion...")
            main()
        except Exception:
            traceback.print_exc(file=log)
            if con_consola:
                traceback.print_exc()
                print(f"Registro: {ruta_log}", file=sys.stderr)
            else:
                import tkinter as tk
                from tkinter import messagebox

                window = tk.Tk()
                window.withdraw()
                messagebox.showerror("No se pudo iniciar", f"Consulta el registro:\n{ruta_log}", parent=window)
                window.destroy()
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(iniciar())
