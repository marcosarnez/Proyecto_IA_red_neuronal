"""Vincula .venv al Python conservado dentro del proyecto, sin instalar paquetes."""
from pathlib import Path
import shutil
import sys


def main():
    raiz = Path(__file__).resolve().parents[1]
    base = raiz / ".runtime" / "python312"
    entorno = raiz / ".venv"
    configuracion = entorno / "pyvenv.cfg"
    if sys.version_info[:2] != (3, 12) or Path(sys.base_prefix).resolve() != base.resolve():
        raise RuntimeError("Ejecuta este archivo con .runtime/python312/python.exe.")
    if not (entorno / "Lib" / "site-packages" / "tensorflow").is_dir():
        raise RuntimeError("Faltan las bibliotecas de .venv. Consulta README.md para restaurarlas.")
    anterior = configuracion.read_text(encoding="utf-8") if configuracion.exists() else ""
    valores = dict(linea.split(" = ", 1) for linea in anterior.splitlines() if " = " in linea)
    correcto = Path(valores.get("home", ".")).resolve() == base.resolve()
    ejecutables = all((entorno / "Scripts" / nombre).is_file() for nombre in ("python.exe", "pythonw.exe"))
    if correcto and ejecutables:
        return
    respaldo = raiz / ".runtime" / "pyvenv.cfg.antes_reparacion"
    if configuracion.exists() and not respaldo.exists():
        shutil.copyfile(configuracion, respaldo)
    # El lanzador de un venv lee home al arrancar. No reemplazar ejecutables
    # existentes: una ventana abierta puede mantener pythonw.exe bloqueado.
    valores.update({
        "home": str(base),
        "include-system-site-packages": "false",
        "version": sys.version.split()[0],
        "executable": str(base / "python.exe"),
        "command": f'"{base / "python.exe"}" -m venv "{entorno}"',
    })
    for nombre in ("python.exe", "pythonw.exe"):
        destino = entorno / "Scripts" / nombre
        if not destino.exists():
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / "Lib" / "venv" / "scripts" / "nt" / nombre, destino)
    configuracion.write_text("".join(f"{clave} = {valor}\n" for clave, valor in valores.items()), encoding="utf-8")
    print("Entorno reparado: ahora utiliza el Python local del proyecto.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"No se pudo preparar el entorno: {error}", file=sys.stderr)
        sys.exit(1)
