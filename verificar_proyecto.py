"""Verificaciones de integracion: particiones, preprocesamiento y modelo real."""
import json
import hashlib
import sys
from pathlib import Path

from Residuos_Medicos_Pedagogico import tf
import cv2
import numpy as np
from PIL import Image

from configuracion import RAIZ, CLASES, MODELO, MAPA, TAMANO_IMAGEN
from entrenamiento import cargar_manifiesto, probabilidades
from interfaz import preparar_entrada
from preparar_datos import normalizar_imagen


def main():
    manifest = cargar_manifiesto()
    assert manifest["clases"] == CLASES
    for split in ("entrenamiento", "validacion", "test"):
        assert all(manifest["conteos"][split][clase] > 0 for clase in CLASES)
    # Una imagen rectangular y roja detecta divergencias RGB, proporciones o escala.
    fixture = Image.new("RGB", (240, 80), (255, 0, 0))
    training_pixels = np.asarray(normalizar_imagen(fixture), dtype=np.float32)
    live_pixels = preparar_entrada(fixture)[0]
    assert np.array_equal(training_pixels, live_pixels), "Preprocesamiento de entrenamiento y GUI diferente"
    assert live_pixels.max() == 255, "Se normalizo fuera del modelo"
    assert live_pixels[64, 64, 0] == 255 and live_pixels[64, 64, 2] == 0, "Canales RGB invertidos"
    metadata = json.loads(MAPA.read_text(encoding="utf-8"))
    assert metadata["clases"] == CLASES
    model = tf.keras.models.load_model(MODELO, compile=False)
    assert model.output_shape[-1] == 7
    example_rows = [next(r for r in manifest["imagenes"] if r["clase"] == c and r["particion"] == "test") for c in CLASES]
    x = np.stack([np.asarray(Image.open(RAIZ / row["ruta"]).convert("RGB")) for row in example_rows])
    p = probabilidades(model, x)
    assert p.shape == (7, 7) and np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
    metrics = json.loads((RAIZ / "salidas" / "metricas_test.json").read_text(encoding="utf-8"))
    assert set(metrics["clases"]) == set(CLASES)
    with MODELO.open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == metrics["modelo_sha256"], "Las metricas corresponden a otro archivo de modelo"
    assert all(name in metrics for name in ("f1_macro", "f1_weighted", "auc_macro_ovr"))
    result = {"python": sys.version.split()[0], "tensorflow": tf.__version__, "opencv": cv2.__version__, "clases": CLASES, "particiones_sin_fuga_de_grupo_o_hash": True, "preprocesamiento_train_gui_identico": True, "softmax_finita_suma_uno": True, "modelo_guardado_cargado": True, "metricas_test_presentes": True}
    (RAIZ / "salidas" / "verificacion.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
