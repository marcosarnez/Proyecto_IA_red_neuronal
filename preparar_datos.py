"""Limpieza, trazabilidad y particion por origen y similitud perceptual."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from configuracion import RAIZ, CLASES, NOMBRES, TAMANO_IMAGEN, SEMILLA
from adquirir_datos import guardar_json, CACHE, META


def normalizar_imagen(image):
    """Mismo RGB, proporcion y relleno que se emplean en la webcam."""
    return ImageOps.pad(image.convert("RGB"), (TAMANO_IMAGEN, TAMANO_IMAGEN), method=Image.Resampling.BILINEAR, color=(127, 127, 127), centering=(0.5, 0.5))


def huella_perceptual(image):
    gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    hashes = []
    for flip in (gray, np.fliplr(gray)):
        for k in range(4):
            block = cv2.dct(np.ascontiguousarray(np.rot90(flip, k)))[:8, :8].flatten()[1:]
            bits = block > np.median(block)
            hashes.append(int("".join("1" if bit else "0" for bit in bits), 2))
    return min(hashes)


class Grupos:
    def __init__(self, n):
        self.parent = list(range(n))

    def raiz(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def unir(self, a, b):
        a, b = self.raiz(a), self.raiz(b)
        if a != b:
            self.parent[b] = a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distancia-phash", type=int, default=7)
    args = parser.parse_args()
    output = RAIZ / "datos" / "preparados"
    output.mkdir(parents=True, exist_ok=True)
    rows, rejected = [], []
    exact = {}
    exclusions_path = RAIZ / "datos" / "exclusiones_manuales.json"
    exclusions = json.loads(exclusions_path.read_text(encoding="utf-8")) if exclusions_path.exists() else []
    for plan in sorted(META.glob("plan_*.json")):
        for record in json.loads(plan.read_text(encoding="utf-8")):
            exclusion = next((e for e in exclusions if e["grupo_origen"] == record["grupo_origen"] and e["clase"] == record["clase"]), None)
            if exclusion:
                rejected.append({"id": record["id"], "motivo": exclusion["motivo"]})
                continue
            source = CACHE / (record.get("original_id", record["id"]) + ".jpg")
            if not source.exists():
                continue
            try:
                with Image.open(source) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    if "bbox" in record:
                        x, y, w, h = record["bbox"]
                        sx, sy = image.width / record["source_width"], image.height / record["source_height"]
                        pad = 0.16
                        box = (max(0, (x-w*pad)*sx), max(0, (y-h*pad)*sy), min(image.width, (x+w*(1+pad))*sx), min(image.height, (y+h*(1+pad))*sy))
                        image = image.crop(box)
                    if min(image.size) < 12:
                        raise ValueError("Objeto demasiado pequeno")
                    image = normalizar_imagen(image)
                    pixels = np.asarray(image)
                    if float(pixels.std()) < 6:
                        raise ValueError("Imagen vacia o casi uniforme")
                    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                    if digest in exact:
                        if exact[digest] != record["clase"]:
                            raise ValueError("Mismos pixeles con etiquetas contradictorias")
                        rejected.append({"id": record["id"], "motivo": "duplicado_exacto"})
                        continue
                    exact[digest] = record["clase"]
                    target = output / (record["id"] + ".png")
                    image.save(target)
                    rows.append({**record, "ruta": str(target.relative_to(RAIZ)).replace("\\", "/"), "sha256_pixels": digest, "phash": huella_perceptual(image)})
            except (OSError, ValueError) as error:
                rejected.append({"id": record["id"], "motivo": str(error)})
    if not rows:
        raise RuntimeError("No hay imagenes disponibles. Ejecuta adquirir_datos.py primero.")
    union = Grupos(len(rows))
    origins = {}
    for i, row in enumerate(rows):
        origin = row["grupo_origen"]
        if origin in origins:
            union.unir(i, origins[origin])
        else:
            origins[origin] = i
    near_pairs = 0
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            if a["clase"] == b["clase"] and (a["phash"] ^ b["phash"]).bit_count() <= args.distancia_phash:
                union.unir(i, j)
                near_pairs += 1
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        row["grupo"] = f"g{union.raiz(i):05d}"
        groups[row["grupo"]].append(i)
    split_for_group = {}
    group_counts = Counter()
    for class_id, clase in enumerate(CLASES):
        candidates = [g for g, indices in groups.items() if any(rows[i]["clase"] == clase for i in indices)]
        group_counts[clase] = len(candidates)
        if len(candidates) < 5:
            raise RuntimeError(f"{clase}: solo {len(candidates)} grupos independientes; agrega datos antes de entrenar.")
        rng = random.Random(SEMILLA + class_id)
        rng.shuffle(candidates)
        # Grupos compartidos entre etiquetas, como una foto con varios objetos,
        # mantienen una sola particion. No se separan sus recortes.
        available = [g for g in candidates if g not in split_for_group]
        n_test = max(1, round(len(available)*0.15))
        n_val = max(1, round(len(available)*0.15))
        for k, group in enumerate(available):
            split_for_group[group] = "test" if k < n_test else "validacion" if k < n_test+n_val else "entrenamiento"
    for row in rows:
        row["particion"] = split_for_group[row["grupo"]]
        row["clase_id"] = CLASES.index(row["clase"])
    counts = {split: {c: sum(r["clase"] == c and r["particion"] == split for r in rows) for c in CLASES} for split in ("entrenamiento", "validacion", "test")}
    for split, classes in counts.items():
        if any(count == 0 for count in classes.values()):
            raise RuntimeError(f"La particion {split} no contiene todas las clases; hace falta mas diversidad.")
    manifest = {"version": 1, "semilla": SEMILLA, "clases": CLASES, "tamano_imagen": TAMANO_IMAGEN, "imagenes": rows, "conteos": counts, "grupos_por_clase": dict(group_counts), "pares_perceptuales_agrupados": near_pairs, "rechazadas": rejected}
    guardar_json(RAIZ / "datos" / "manifiesto.json", manifest)
    audit = {"imagenes": len(rows), "grupos": len(groups), "conteos": counts, "grupos_por_clase": dict(group_counts), "rechazadas": len(rejected), "distancia_phash": args.distancia_phash, "grupos_compartidos_entre_particiones": 0, "advertencia": "La agrupacion por nombre y pHash reduce fugas evidentes; no demuestra independencia de todos los objetos fisicos de origen."}
    guardar_json(RAIZ / "salidas" / "auditoria_datos.json", audit)
    contact = Image.new("RGB", (1000, 170*len(CLASES)), "#eef2f6")
    draw = ImageDraw.Draw(contact)
    for i, clase in enumerate(CLASES):
        draw.text((10, i*170+4), f"{NOMBRES[i]} | {group_counts[clase]} grupos", fill="black")
        examples = [r for r in rows if r["clase"] == clase]
        random.Random(SEMILLA).shuffle(examples)
        for j, row in enumerate(examples[:7]):
            with Image.open(RAIZ / row["ruta"]) as photo:
                contact.paste(photo, (j*142+6, i*170+25))
    contact.save(RAIZ / "salidas" / "auditoria_visual.jpg", quality=90)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
