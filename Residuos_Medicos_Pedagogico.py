"""Clasificacion multiclase de residuos: CNN explicita y aplicacion de escritorio.

Adaptacion teorica del ejemplo Con_Sin_Lentes_Pedagogico.py visible en la
diapositiva 750. La diapositiva solo muestra parte del codigo del docente;
esta es una implementacion propia de los conceptos y requisitos solicitados.
"""
# SECCION 0 - IMPORTS: las mismas familias de librerias del ejemplo docente.
import os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))
from entorno import configurar_tk
configurar_tk()

import argparse
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from configuracion import CLASES, TAMANO_IMAGEN, SEMILLA

# SECCION 1 - REPRODUCIBILIDAD Y RECURSOS.
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.keras.utils.set_random_seed(SEMILLA)


def construir_modelo():
    """Una salida por clase; el orden se guarda junto con el modelo .keras."""
    modelo = Sequential([
        Input(shape=(TAMANO_IMAGEN, TAMANO_IMAGEN, 3), name="imagen_rgb"),
        # Pixeles 0..255 -> 0..1. No normalizar otra vez en la webcam.
        Rescaling(1./255, name="normalizacion"),
        # Solo actuan al entrenar; no alteran validacion, test o webcam.
        RandomFlip("horizontal", seed=SEMILLA),
        RandomRotation(0.08, fill_mode="reflect", seed=SEMILLA+1),
        RandomZoom(0.12, fill_mode="reflect", seed=SEMILLA+2),
        RandomTranslation(0.10, 0.10, fill_mode="reflect", seed=SEMILLA+3),
        RandomContrast(0.15, seed=SEMILLA+4),
        # CONV aprende filtros. SAME conserva dimensiones con stride 1.
        # ReLU conserva gradiente para entradas positivas y agrega no linealidad.
        Conv2D(16, 3, padding="same", activation="relu", name="conv_bordes"),
        # POOL reduce resolucion y aporta invariancia local a desplazamientos.
        MaxPooling2D(2, name="pool_1"),
        Conv2D(32, 3, padding="same", activation="relu", name="conv_texturas"),
        MaxPooling2D(2, name="pool_2"),
        Conv2D(64, 3, padding="same", activation="relu", name="conv_formas"),
        MaxPooling2D(2, name="pool_3"),
        Flatten(name="aplanar"),
        Dense(64, activation="relu", name="clasificador"),
        # Dropout reduce coadaptacion y ayuda a mitigar alta varianza.
        Dropout(0.40, seed=SEMILLA, name="regularizacion_dropout"),
        # Softmax produce probabilidades multiclase cuya suma es 1.
        Dense(len(CLASES), activation="softmax", name="probabilidades"),
    ], name="CNN_Residuos_Pedagogica")
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return modelo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrenar", action="store_true")
    parser.add_argument("--epocas", type=int, default=24)
    parser.add_argument("--prueba-corta", action="store_true")
    parser.add_argument("--evaluar", action="store_true")
    parser.add_argument("--camara", type=int, default=0)
    parser.add_argument("--autocerrar", type=int, default=0, help="Segundos para una prueba automatica de apertura de GUI")
    parser.add_argument("--imagen", type=str, default=None)
    args = parser.parse_args()
    if args.entrenar:
        from entrenamiento import entrenar
        entrenar(args, construir_modelo())
    elif args.evaluar:
        from entrenamiento import evaluar_guardado
        evaluar_guardado()
    else:
        from interfaz import ejecutar
        ejecutar(args)


if __name__ == "__main__":
    main()
