"""Prueba para la defensa: arquitectura visible, gradientes y pesos que aprenden.

Se construye una red nueva en memoria. El modelo de la aplicacion no se entrena
ni se sobrescribe. Teachable Machine tambien utiliza redes neuronales; aqui se
demuestra la implementacion y el entrenamiento accesibles en este proyecto.
"""
import hashlib
import json

print("Cargando las librerias locales. La primera apertura puede tardar unos segundos...", flush=True)

import numpy as np
from PIL import Image

from Residuos_Medicos_Pedagogico import construir_modelo, tf
from configuracion import RAIZ, MODELO, MAPA, CLASES


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def firma(model):
    return [(type(layer).__name__, getattr(layer, "filters", None), getattr(layer, "units", None)) for layer in model.layers]


def main():
    print("DEMOSTRACION DE LA RED NEURONAL", flush=True)
    print("1. Se construye una CNN nueva desde Sequential y sus capas.", flush=True)
    model = construir_modelo()
    model.summary()
    active_hash = sha256(MODELO)
    active = tf.keras.models.load_model(MODELO, compile=False)
    same_architecture = firma(active) == firma(model) and active.count_params() == model.count_params()
    metadata = json.loads(MAPA.read_text(encoding="utf-8"))
    checkpoint = RAIZ / "salidas" / metadata["ejecucion"] / "modelo_residuos.keras"
    same_checkpoint = sha256(checkpoint) == active_hash
    assert same_architecture and same_checkpoint
    print("Modelo activo: misma arquitectura y mismo hash que el checkpoint del entrenamiento local.", flush=True)
    manifest = json.loads((RAIZ / "datos" / "manifiesto.json").read_text(encoding="utf-8"))
    examples = []
    for name in CLASES:
        examples.extend([row for row in manifest["imagenes"] if row["clase"] == name and row["particion"] == "entrenamiento"][:2])
    x = np.stack([np.asarray(Image.open(RAIZ / row["ruta"]).convert("RGB"), dtype=np.float32) for row in examples])
    y = tf.one_hot([row["clase_id"] for row in examples], len(CLASES))
    convolution = model.get_layer("conv_bordes")
    initial = convolution.get_weights()[0].copy()
    print("2. Retropropagacion explicita con GradientTape y actualizacion con Adam.", flush=True)
    steps = []
    for step in range(3):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(gradient is not None and bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients)
        norm = float(tf.linalg.global_norm(gradients))
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        difference = float(np.max(np.abs(convolution.get_weights()[0] - initial)))
        row = {"paso": step+1, "cross_entropy": float(loss), "norma_gradientes": norm, "cambio_maximo_kernel_desde_inicio": difference}
        steps.append(row)
        print(f"   Paso {step+1}: loss={float(loss):.6f}; norma del gradiente={norm:.6f}; cambio de pesos={difference:.8f}", flush=True)
    final = convolution.get_weights()[0]
    probabilities = np.asarray(model(x[:1], training=False))[0]
    assert not np.array_equal(initial, final)
    assert np.isclose(probabilities.sum(), 1, atol=1e-5)
    unchanged = sha256(MODELO) == active_hash
    assert unchanged
    evidence = {"parametros_entrenables": int(sum(np.prod(variable.shape) for variable in model.trainable_variables)), "primera_convolucion_parametros": convolution.count_params(), "formula_primera_convolucion": "3*3*3*16 + 16 = 448", "arquitectura_compatible_con_modelo_activo": same_architecture, "modelo_activo_coincide_con_checkpoint_local": same_checkpoint, "modelo_activo_sha256": active_hash, "ejemplos_entrenamiento_en_demostracion": len(examples), "pesos_antes_muestra": initial.flatten()[:6].tolist(), "pesos_despues_muestra": final.flatten()[:6].tolist(), "pasos": steps, "probabilidades_ejemplo": probabilities.tolist(), "suma_probabilidades": float(probabilities.sum()), "pesos_de_la_red_nueva_cambiaron": True, "archivo_modelo_aplicacion_sin_cambios": unchanged}
    target = RAIZ / "salidas" / "demostracion_red.json"
    target.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print("3. Los pesos cambiaron; softmax suma", round(float(probabilities.sum()), 6), flush=True)
    print("El archivo del modelo de la aplicacion permanece sin cambios.", flush=True)
    print("Evidencia guardada en:", target, flush=True)
    print("Tres pasos muestran el mecanismo de aprendizaje; no evalúan la calidad final del clasificador.", flush=True)


if __name__ == "__main__":
    main()
