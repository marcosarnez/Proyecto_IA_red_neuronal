"""Entrenamiento, metricas multiclase y evaluacion final con test reservado."""
import csv
from datetime import datetime
import hashlib
import json
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from configuracion import RAIZ, CLASES, NOMBRES, SEMILLA, TAMANO_IMAGEN, MODELO, MAPA
from adquirir_datos import guardar_json


def cargar_manifiesto():
    data = json.loads((RAIZ / "datos" / "manifiesto.json").read_text(encoding="utf-8"))
    if data["clases"] != CLASES or data["tamano_imagen"] != TAMANO_IMAGEN:
        raise ValueError("El manifiesto no coincide con la configuracion del modelo")
    by_group, by_hash = {}, {}
    for row in data["imagenes"]:
        for field, index in (("grupo", by_group), ("sha256_pixels", by_hash)):
            key = row[field]
            if key in index and index[key] != row["particion"]:
                raise ValueError(f"Fuga detectada: {field} aparece en varias particiones")
            index[key] = row["particion"]
    return data


def cargar_particion(manifest, split):
    rows = [r for r in manifest["imagenes"] if r["particion"] == split]
    x = np.stack([np.asarray(Image.open(RAIZ / r["ruta"]).convert("RGB"), dtype=np.uint8) for r in rows])
    y = np.asarray([r["clase_id"] for r in rows], dtype=np.int32)
    if set(y.tolist()) != set(range(len(CLASES))):
        raise ValueError(f"Faltan clases en {split}")
    return x, y, rows


def probabilidades(modelo, x, batch=32):
    predictions = np.concatenate([np.asarray(modelo(x[start:start+batch], training=False)) for start in range(0, len(x), batch)])
    if not np.isfinite(predictions).all() or not np.allclose(predictions.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Salida softmax invalida")
    return predictions


def metricas(y, p):
    return {"accuracy": float(np.mean(p.argmax(axis=1) == y)), "f1_macro": float(f1_score(y, p.argmax(axis=1), average="macro", zero_division=0)), "f1_weighted": float(f1_score(y, p.argmax(axis=1), average="weighted", zero_division=0)), "auc_macro_ovr": float(roc_auc_score(np.eye(len(CLASES))[y], p, average="macro"))}


def dataset(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, np.eye(len(CLASES), dtype=np.float32)[y]))
    if training:
        ds = ds.shuffle(len(x), seed=SEMILLA, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 2
    return ds.batch(24).with_options(options).prefetch(1)


class MetricasValidacion(tf.keras.callbacks.Callback):
    def __init__(self, x, y, output):
        super().__init__()
        self.x, self.y, self.output = x, y, output
        self.history = []
        self.start = time.monotonic()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        scores = metricas(self.y, probabilidades(self.model, self.x))
        logs["val_f1_macro"] = scores["f1_macro"]
        logs["val_auc_macro_ovr"] = scores["auc_macro_ovr"]
        row = {"epoca": epoch+1, "train_loss": float(logs["loss"]), "val_loss": float(logs["val_loss"]), "train_accuracy": float(logs["accuracy"]), "val_accuracy": float(logs["val_accuracy"]), "val_f1_macro": scores["f1_macro"], "val_auc_macro_ovr": scores["auc_macro_ovr"], "segundos_acumulados": round(time.monotonic()-self.start, 2)}
        self.history.append(row)
        guardar_json(self.output / "historial.json", self.history)
        print(f"    F1 macro val={scores['f1_macro']:.3f} | AUC-ROC OVR val={scores['auc_macro_ovr']:.3f}", flush=True)


def graficar_historial(rows, output):
    epochs = [r["epoca"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="Entrenamiento")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], label="Validación")
    axes[0].set(title="Cross-Entropy: aprendizaje y generalización", ylabel="Pérdida")
    axes[1].plot(epochs, [r["train_accuracy"] for r in rows], label="Entrenamiento")
    axes[1].plot(epochs, [r["val_accuracy"] for r in rows], label="Validación")
    axes[1].set(title="Exactitud", ylabel="Accuracy", ylim=(0, 1.03))
    axes[2].plot(epochs, [r["val_f1_macro"] for r in rows], label="F1 macro")
    axes[2].plot(epochs, [r["val_auc_macro_ovr"] for r in rows], label="AUC macro OVR")
    axes[2].set(title="Validación: métricas por clase", ylim=(0, 1.03))
    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "graficos_entrenamiento.png", dpi=160)
    plt.close(fig)
    with (output / "historial.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluar(modelo, manifest, output):
    x, y, rows = cargar_particion(manifest, "test")
    p = probabilidades(modelo, x)
    result = metricas(y, p)
    result["imagenes_test"] = len(y)
    result["reporte_por_clase"] = classification_report(y, p.argmax(axis=1), labels=list(range(len(CLASES))), target_names=NOMBRES, output_dict=True, zero_division=0)
    result["clases"] = CLASES
    result["modelo_sha256"] = hashlib.file_digest((output / "modelo_residuos.keras").open("rb"), "sha256").hexdigest()
    guardar_json(output / "metricas_test.json", result)
    report = classification_report(y, p.argmax(axis=1), target_names=NOMBRES, zero_division=0, digits=3)
    (output / "reporte_clasificacion.txt").write_text(report, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix(y, p.argmax(axis=1)), display_labels=NOMBRES).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35)
    ax.set_title("Test reservado: matriz de confusión")
    fig.tight_layout()
    fig.savefig(output / "matriz_confusion.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(NOMBRES):
        fpr, tpr, _ = roc_curve((y == i).astype(int), p[:, i])
        ax.plot(fpr, tpr, label=f"{name} ({auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Sensibilidad", title="ROC multiclase: una clase frente al resto", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "curvas_roc.png", dpi=160)
    plt.close(fig)
    # Casos de respaldo para la defensa: se conservan tambien los errores.
    prediction_rows = [{"id": row["id"], "ruta": row["ruta"], "real": CLASES[int(label)], "predicha": CLASES[int(prob.argmax())], "probabilidad": float(prob.max())} for row, label, prob in zip(rows, y, p)]
    guardar_json(output / "predicciones_test.json", prediction_rows)
    print(json.dumps({k: v for k, v in result.items() if k != "reporte_por_clase"}, ensure_ascii=False, indent=2), flush=True)
    return result


def entrenar(args, modelo):
    manifest = cargar_manifiesto()
    x_train, y_train, _ = cargar_particion(manifest, "entrenamiento")
    x_val, y_val, _ = cargar_particion(manifest, "validacion")
    if args.prueba_corta:
        indices = np.concatenate([np.flatnonzero(y_train == i)[:12] for i in range(len(CLASES))])
        x_train, y_train = x_train[indices], y_train[indices]
        epochs = min(args.epocas, 2)
        output = RAIZ / "salidas" / "prueba_corta"
    else:
        epochs = args.epocas
        output = RAIZ / "salidas" / ("ejecucion_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASES)), y=y_train)
    guardar_json(output / "configuracion_entrenamiento.json", {"epocas_maximas": epochs, "clases": CLASES, "pesos_clases": dict(zip(CLASES, weights.tolist())), "imagenes_train": len(y_train), "imagenes_validacion": len(y_val), "prueba_corta": args.prueba_corta, "semilla": SEMILLA})
    with (output / "arquitectura.txt").open("w", encoding="utf-8") as file:
        modelo.summary(print_fn=lambda line: file.write(line + "\n"))
    callback = MetricasValidacion(x_val, y_val, output)
    callbacks = [callback, tf.keras.callbacks.ModelCheckpoint(str(output / "modelo_residuos.keras"), monitor="val_loss", save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, min_delta=0.001, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=0.00002)]
    print(f"Entrenando {len(y_train)} ejemplos; validando {len(y_val)}; hasta {epochs} epocas. Salida: {output}", flush=True)
    modelo.fit(dataset(x_train, y_train, training=True), validation_data=dataset(x_val, y_val), epochs=epochs, callbacks=callbacks, class_weight={i: float(w) for i, w in enumerate(weights)}, verbose=2)
    graficar_historial(callback.history, output)
    best = tf.keras.models.load_model(output / "modelo_residuos.keras", compile=False)
    validation = metricas(y_val, probabilidades(best, x_val))
    guardar_json(output / "metricas_validacion.json", validation)
    if args.prueba_corta:
        print("Prueba corta completa. No se uso test ni se activo este modelo para la GUI.", flush=True)
        return
    evaluar(best, manifest, output)
    metadata = {"clases": CLASES, "nombres": NOMBRES, "tamano_imagen": TAMANO_IMAGEN, "normalizacion_en_modelo": True, "umbral_sugerido": 0.65, "margen_minimo": 0.12, "ejecucion": output.name, "validacion": validation}
    guardar_json(output / "clases.json", metadata)
    # Se activa solamente un entrenamiento completo, con sus metricas y etiquetas.
    for name in ("modelo_residuos.keras", "clases.json", "metricas_test.json", "metricas_validacion.json", "graficos_entrenamiento.png", "matriz_confusion.png", "curvas_roc.png", "reporte_clasificacion.txt", "predicciones_test.json", "historial.csv", "historial.json", "arquitectura.txt"):
        shutil.copy2(output / name, RAIZ / "salidas" / name)
    print("Modelo final y resultados activados en salidas/", flush=True)


def evaluar_guardado():
    if not MODELO.exists():
        raise FileNotFoundError("Primero entrena y guarda un modelo completo")
    model = tf.keras.models.load_model(MODELO, compile=False)
    evaluar(model, cargar_manifiesto(), RAIZ / "salidas")
