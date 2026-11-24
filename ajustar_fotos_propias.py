"""Ajusta la CNN existente con las fotos de la demostración y conserva el original."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from configuracion import CLASES, MAPA, MODELO, RAIZ, TAMANO_IMAGEN


FOTOS = RAIZ / "datos" / "fotos_defensa_etiquetas.json"
SALIDA = RAIZ / "salidas" / "ajuste_fotos_propias"


def preparar(image):
    image = ImageOps.pad(
        image.convert("RGB"),
        (TAMANO_IMAGEN, TAMANO_IMAGEN),
        method=Image.Resampling.BILINEAR,
        color=(127, 127, 127),
    )
    return np.asarray(image, dtype=np.uint8)


def recortar_objeto(image):
    """Aproxima el encuadre central que se muestra en la webcam."""
    width, height = image.size
    side = int(min(width * 0.9, height * 0.68))
    center_x = width // 2
    center_y = int(height * 0.61)
    return image.crop((center_x - side // 2, center_y - side // 2,
                       center_x + side // 2, center_y + side // 2))


def cargar_fotos():
    manifest = json.loads(FOTOS.read_text(encoding="utf-8"))
    images, cropped, labels, names = [], [], [], []
    for entry in manifest["imagenes"]:
        path = RAIZ / entry["archivo"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"La foto cambió o falta: {path}")
        with Image.open(path) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
            images.append(preparar(photo))
            cropped.append(preparar(recortar_objeto(photo)))
        labels.append(CLASES.index(entry["clase"]))
        names.append(entry["archivo"])
    return np.stack(images), np.stack(cropped), np.asarray(labels), names


def predicciones(model, images):
    batches = []
    for first in range(0, len(images), 24):
        batches.append(np.asarray(model(images[first:first + 24], training=False)))
    return np.concatenate(batches)


def aciertos(model, images, labels):
    scores = predicciones(model, images)
    ranked = np.sort(scores, axis=1)
    accepted = (scores.max(axis=1) >= 0.65) & (ranked[:, -1] - ranked[:, -2] >= 0.12)
    correct = scores.argmax(axis=1) == labels
    return int((accepted & correct).sum()), len(labels), int(correct.sum())


def evaluar_original(model):
    manifest = json.loads((RAIZ / "datos" / "manifiesto.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest["imagenes"] if row["particion"] == "validacion"]
    images = np.stack([np.asarray(Image.open(RAIZ / row["ruta"]).convert("RGB")) for row in rows])
    labels = np.asarray([row["clase_id"] for row in rows])
    scores = predicciones(model, images)
    return float(np.mean(scores.argmax(axis=1) == labels))


def muestras_originales(por_clase):
    manifest = json.loads((RAIZ / "datos" / "manifiesto.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest["imagenes"] if row["particion"] == "entrenamiento"]
    rng = np.random.default_rng(42)
    chosen = []
    for class_id in range(len(CLASES)):
        indices = [index for index, row in enumerate(rows) if row["clase_id"] == class_id]
        chosen.extend(rng.choice(indices, size=min(por_clase, len(indices)), replace=False))
    images = np.stack([np.asarray(Image.open(RAIZ / rows[index]["ruta"]).convert("RGB")) for index in chosen])
    labels = np.asarray([rows[index]["clase_id"] for index in chosen])
    return images, labels


def entrenar(epocas, activar, solo_cabeza, tasa, repeticiones, originales_por_clase):
    SALIDA.mkdir(parents=True, exist_ok=True)
    original = RAIZ / "salidas" / "modelo_residuos_original.keras"
    fuente = original if original.exists() else MODELO
    model = tf.keras.models.load_model(fuente, compile=False)
    whole, cropped, labels, names = cargar_fotos()
    train_indices, held_indices = [], []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        held_indices.extend(indices[4::5])
        train_indices.extend(index for index in indices if index not in held_indices)
    train_indices = np.asarray(train_indices)
    held_indices = np.asarray(held_indices)

    before_full = aciertos(model, whole, labels)
    before_crop = aciertos(model, cropped, labels)
    old_validation = evaluar_original(model)
    print(f"Antes, fotos completas aceptadas: {before_full[0]}/{before_full[1]}", flush=True)
    print(f"Antes, recortes aceptados: {before_crop[0]}/{before_crop[1]}", flush=True)
    print(f"Antes, validación original: {old_validation:.3f}", flush=True)

    original_images, original_labels = muestras_originales(originales_por_clase)
    own_images = np.concatenate((whole[train_indices], cropped[train_indices]))
    own_labels = np.tile(labels[train_indices], 2)
    train_images = np.concatenate([original_images] + [own_images] * repeticiones)
    train_labels = np.concatenate([original_labels] + [own_labels] * repeticiones)
    one_hot = np.eye(len(CLASES), dtype=np.float32)[train_labels]
    dataset = tf.data.Dataset.from_tensor_slices((train_images, one_hot))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    dataset = dataset.shuffle(len(train_images), seed=42).batch(24).with_options(options).prefetch(1)

    if solo_cabeza:
        for layer in model.layers:
            layer.trainable = layer.name in ("clasificador", "probabilidades")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tasa),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    best_path = SALIDA / ("modelo_cabeza.keras" if solo_cabeza else "modelo_candidato.keras")
    best_score = -1.0
    history = []
    for epoch in range(epocas):
        result = model.fit(dataset, epochs=1, verbose=0)
        own_full = aciertos(model, whole[held_indices], labels[held_indices])
        own_crop = aciertos(model, cropped[held_indices], labels[held_indices])
        validation = evaluar_original(model)
        score = (own_full[0] + own_crop[0]) / (2 * len(held_indices))
        score -= max(0.0, old_validation - validation) * 0.5
        row = {"epoca": epoch + 1, "perdida": float(result.history["loss"][0]),
               "fotos_reservadas_completas": own_full[0], "fotos_reservadas_recortadas": own_crop[0],
               "fotos_reservadas_total": len(held_indices), "validacion_original": validation,
               "puntuacion_seleccion": score}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score = score
            model.save(best_path)

    best = tf.keras.models.load_model(best_path, compile=False)
    after_full = aciertos(best, whole, labels)
    after_crop = aciertos(best, cropped, labels)
    final_validation = evaluar_original(best)
    report = {"fotos_propias": len(labels), "fotos_reservadas": len(held_indices),
              "solo_cabeza": solo_cabeza, "tasa": tasa,
              "fotos_propias_antes": {"completas": before_full, "recortadas": before_crop},
              "fotos_propias_despues": {"completas": after_full, "recortadas": after_crop},
              "validacion_original_antes": old_validation,
              "validacion_original_despues": final_validation,
              "epocas": history,
              "nota": "Las fotos reservadas muestran los mismos objetos y fondo que las fotos de ajuste; no miden generalización independiente."}
    report_path = SALIDA / ("resultado_cabeza.json" if solo_cabeza else "resultado.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "epocas"}, ensure_ascii=False, indent=2), flush=True)

    if activar:
        if not original.exists():
            shutil.copy2(MODELO, original)
        next_path = RAIZ / "salidas" / "modelo_residuos_nuevo.keras"
        shutil.copy2(best_path, next_path)
        os.replace(next_path, MODELO)
        metadata = json.loads(MAPA.read_text(encoding="utf-8"))
        metadata["ajuste_fotos_propias"] = {
            "cantidad": len(labels), "resultado": str(report_path.relative_to(RAIZ)),
            "metricas_previas": "salidas/ejecucion_20261117_221647/metricas_test.json"}
        MAPA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Modelo ajustado activado. Original conservado como modelo_residuos_original.keras.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epocas", type=int, default=8)
    parser.add_argument("--activar", action="store_true")
    parser.add_argument("--solo-cabeza", action="store_true")
    parser.add_argument("--tasa", type=float, default=0.00005)
    parser.add_argument("--repeticiones", type=int, default=2)
    parser.add_argument("--originales-por-clase", type=int, default=35)
    args = parser.parse_args()
    entrenar(args.epocas, args.activar, args.solo_cabeza, args.tasa,
            args.repeticiones, args.originales_por_clase)
