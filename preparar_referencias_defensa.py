"""Guarda rasgos extraídos por la CNN ajustada de las fotos propias."""

import hashlib
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from ajustar_fotos_propias import cargar_fotos
from configuracion import CLASES, RAIZ


def main():
    output = RAIZ / "salidas" / "ajuste_fotos_propias"
    model_path = output / "modelo_candidato.keras"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    whole, cropped, labels, names = cargar_fotos()
    images = np.concatenate((whole, cropped))
    model = tf.keras.models.load_model(model_path, compile=False)
    extractor = tf.keras.Model(model.inputs, model.get_layer("clasificador").output)
    parts = []
    for start in range(0, len(images), 16):
        parts.append(np.asarray(extractor([images[start:start + 16]], training=False)))
    features = np.concatenate(parts)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    np.savez_compressed(output / "referencias.npz", features=features,
                        labels=np.tile(labels, 2))
    metadata = {
        "clases": CLASES,
        "fotos_propias": len(names),
        "variantes_por_foto": 2,
        "modelo_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "origen": "Fotos proporcionadas por Marcos para su demostración",
        "nota": "La comparación por similitud usa rasgos de la CNN. El porcentaje de similitud no es probabilidad calibrada.",
    }
    (output / "referencias.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Preparadas {len(features)} referencias de {len(names)} fotos.")


if __name__ == "__main__":
    main()
