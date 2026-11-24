"""Descargas publicas individuales, reanudables y limitadas por lote y bytes.

No descarga el ZIP completo de Kaggle. Solo conserva imagenes, metadatos y
su procedencia. Los datos se separan posteriormente por grupos de origen.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import random
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from configuracion import RAIZ, CLASES, NOMBRES, SEMILLA

REF = "mario78/biomedical-waste-classification-images-dataset"
CARPETAS = {
    "jeringas": ["used_syringes"],
    "guantes": ["used_medical_gloves"],
    "mascarillas": ["used_masks"],
    "gasas_vendas": ["blood_soaked_bandages"],
    "frascos_ampollas": ["ampoules_full", "ampuoles_broken", "vaccine_or_medicine_vials"],
    "medicamentos": ["expired_tablets"],
}
CACHE = RAIZ / "datos" / "originales"
META = RAIZ / "datos" / "metadatos"


def guardar_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def descargar(url, max_bytes=3_000_000):
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "BiomedicalWaste-Academic/1.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                length = int(response.headers.get("Content-Length", 0))
                if length > max_bytes:
                    raise ValueError(f"Respuesta supera el limite de {max_bytes} bytes")
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("Descarga interrumpida por limite de bytes")
            return data
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def plan_kaggle(por_clase):
    files = json.loads((META / "kaggle_archivos.json").read_text(encoding="utf-8"))
    records = []
    rng = random.Random(SEMILLA)
    for clase, folders in CARPETAS.items():
        queues = []
        for folder in folders:
            candidates = [f for f in files if f["name"].startswith(f"Dataset/Raw/{folder}/") and f["name"].lower().endswith((".jpg", ".jpeg", ".png")) and int(f["totalBytes"]) <= 600_000]
            rng.shuffle(candidates)
            queues.append(candidates)
        chosen = []
        while len(chosen) < por_clase and any(queues):
            for queue in queues:
                if queue and len(chosen) < por_clase:
                    chosen.append(queue.pop())
        for item in chosen:
            name = item["name"]
            identity = hashlib.sha256(name.encode()).hexdigest()[:20]
            folder = Path(name).parent.name
            stem = Path(name).stem.split(".rf.")[0].casefold()
            records.append({"id": "kg_" + identity, "clase": clase, "fuente": "kaggle_mario78", "archivo_fuente": name, "grupo_origen": f"kaggle:{folder}:{stem}", "bytes_estimados": int(item["totalBytes"]), "licencia": "MIT (ficha Kaggle)", "url_fuente": "https://www.kaggle.com/datasets/" + REF, "url": "https://www.kaggle.com/api/v1/datasets/download/" + REF + "/" + urllib.parse.quote(name, safe="")})
    # Intercalar clases permite revisar todas con un lote pequeno.
    grouped = {c: [r for r in records if r["clase"] == c] for c in CLASES}
    return [grouped[c][i] for i in range(max(map(len, grouped.values()))) for c in CLASES if i < len(grouped[c])]


def obtener(record):
    target = CACHE / (record.get("original_id", record["id"]) + ".jpg")
    if target.exists():
        return record, 0
    data = descargar(record["url"], max_bytes=1_200_000)
    if data.startswith(b"PK"):
        archive = zipfile.ZipFile(io.BytesIO(data))
        members = archive.infolist()
        if len(members) != 1 or members[0].file_size > 1_200_000:
            raise ValueError("ZIP individual inesperado")
        data = archive.read(members[0])
    with Image.open(io.BytesIO(data)) as original:
        original.load()
        image = ImageOps.exif_transpose(original).convert("RGB")
        if min(image.size) < 32:
            raise ValueError("Imagen demasiado pequena")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        temporary = target.with_suffix(".tmp")
        image.save(temporary, format="JPEG", quality=94)
        temporary.replace(target)
    return record, len(data)


def vista_previa(records):
    output = RAIZ / "salidas" / "muestras_dataset.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (960, 230 * 6), "#e7edf2")
    draw = ImageDraw.Draw(sheet)
    for row, clase in enumerate(CLASES[:6]):
        examples = [r for r in records if r["clase"] == clase and (CACHE / (r["id"] + ".jpg")).exists()][:4]
        draw.text((12, row * 230 + 6), NOMBRES[CLASES.index(clase)], fill="black")
        for col, record in enumerate(examples):
            with Image.open(CACHE / (record["id"] + ".jpg")) as photo:
                tile = ImageOps.pad(photo, (230, 195), color="white")
                sheet.paste(tile, (col * 240 + 5, row * 230 + 28))
    sheet.save(output, quality=88)
    return output


def adquirir_trashnet(por_clase):
    archive_path = META / "trashnet_dataset-resized.zip"
    if not archive_path.exists():
        print("TrashNet: descarga acotada del ZIP de 42.8 MB", flush=True)
        data = descargar("https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip", max_bytes=44_000_000)
        archive_path.write_bytes(data)
    records = []
    rng = random.Random(SEMILLA)
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [name for name in archive.namelist() if "/paper/" in name or "/cardboard/" in name]
        candidates = [name for name in candidates if name.lower().endswith((".jpg", ".jpeg", ".png")) and "__MACOSX" not in name]
        rng.shuffle(candidates)
        for name in candidates[:por_clase]:
            identity = "tn_" + hashlib.sha256(name.encode()).hexdigest()[:20]
            target = CACHE / (identity + ".jpg")
            if not target.exists():
                member = archive.getinfo(name)
                if member.file_size > 2_000_000:
                    raise ValueError("Imagen de TrashNet inesperadamente grande")
                image = Image.open(io.BytesIO(archive.read(name))).convert("RGB")
                image.save(target, quality=94)
            records.append({"id": identity, "clase": "papel_carton", "fuente": "trashnet", "archivo_fuente": name, "grupo_origen": "trashnet:" + name, "licencia": "MIT (repositorio original)", "url_fuente": "https://github.com/garythung/trashnet"})
    guardar_json(META / "plan_trashnet.json", records)
    print(f"TrashNet preparado: {len(records)} imagenes de papel/carton", flush=True)


def plan_medbin(por_clase):
    commit = "fee9be8d9aeea4d9ea05384be648ea8b178d5262"
    base = f"https://raw.githubusercontent.com/Wayne3918/MedbinNet/{commit}/MMdetection"
    target_labels = {"jeringas": {"syringe"}, "guantes": {"medical_gloves"}, "mascarillas": {"Mask", "N95"}, "gasas_vendas": {"gauze", "bandage"}, "frascos_ampollas": {"glass_bottle"}, "medicamentos": {"pill", "capsule"}}
    candidates = {clase: {} for clase in target_labels}
    for split in ("train", "valid"):
        path = META / f"medbin_{split}.json"
        if not path.exists():
            path.write_bytes(descargar(base + f"/{split}/_annotations.coco.json", max_bytes=9_000_000))
        data = json.loads(path.read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in data["categories"]}
        images = {image["id"]: image for image in data["images"]}
        for annotation in data["annotations"]:
            label = names[annotation["category_id"]]
            clase = next((c for c, labels in target_labels.items() if label in labels), None)
            if clase is None:
                continue
            x, y, w, h = annotation["bbox"]
            if min(w, h) < 15:
                continue
            image = images[annotation["image_id"]]
            filename = image["file_name"]
            origin = filename.split(".rf.")[0].casefold()
            remote = f"{split}/{filename}"
            original_id = "mb_" + hashlib.sha256(remote.encode()).hexdigest()[:20]
            identity = "crop_" + hashlib.sha256((remote + str(annotation["id"])).encode()).hexdigest()[:20]
            record = {"id": identity, "original_id": original_id, "clase": clase, "fuente": "medbin", "archivo_fuente": remote, "grupo_origen": "medbin:" + origin, "bbox": [x, y, w, h], "source_width": image["width"], "source_height": image["height"], "bytes_estimados": 60_000, "licencia": "CC BY 4.0", "url_fuente": "https://github.com/Wayne3918/MedbinNet", "url": base + "/" + split + "/" + urllib.parse.quote(filename, safe="")}
            prior = candidates[clase].get(origin)
            if prior is None or w*h > prior["bbox"][2]*prior["bbox"][3]:
                candidates[clase][origin] = record
    grouped = {}
    for i, clase in enumerate(CLASES):
        items = list(candidates.get(clase, {}).values())
        random.Random(SEMILLA+i).shuffle(items)
        grouped[clase] = items[:por_clase]
    return [grouped[c][i] for i in range(max(map(len, grouped.values()))) for c in CLASES if i < len(grouped[c])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--por-clase", type=int, default=160)
    parser.add_argument("--max-descargas", type=int, default=48)
    parser.add_argument("--presupuesto-mb", type=float, default=40)
    parser.add_argument("--hilos", type=int, default=4)
    parser.add_argument("--trashnet", action="store_true")
    parser.add_argument("--medbin", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.trashnet:
        adquirir_trashnet(args.por_clase)
        return
    records = plan_medbin(args.por_clase) if args.medbin else plan_kaggle(args.por_clase)
    guardar_json(META / ("plan_medbin.json" if args.medbin else "plan_kaggle.json"), records)
    pending = []
    seen = set()
    for record in records:
        identity = record.get("original_id", record["id"])
        if not (CACHE / (identity + ".jpg")).exists() and identity not in seen:
            pending.append(record)
            seen.add(identity)
    selected = []
    estimate = 0
    for record in pending:
        if len(selected) >= args.max_descargas:
            break
        if estimate + record["bytes_estimados"] > args.presupuesto_mb * 1_000_000:
            continue
        selected.append(record)
        estimate += record["bytes_estimados"]
    print(json.dumps({"plan": len(records), "ya_descargadas": len(records)-len(pending), "lote": len(selected), "mb_estimados": round(estimate/1e6, 2)}), flush=True)
    errors = []
    total = 0
    with ThreadPoolExecutor(max_workers=max(1, min(args.hilos, 6))) as executor:
        futures = {executor.submit(obtener, r): r for r in selected}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                _, size = future.result()
                total += size
            except Exception as error:
                errors.append({"id": futures[future]["id"], "error": str(error)})
            if i % 12 == 0 or i == len(selected):
                print(f"Lote {i}/{len(selected)}; {total/1e6:.2f} MB; errores {len(errors)}", flush=True)
    guardar_json(META / "errores_descarga.json", errors)
    counts = {c: sum(r["clase"] == c and (CACHE / (r.get("original_id", r["id"]) + ".jpg")).exists() for r in records) for c in CLASES}
    print(json.dumps(counts), flush=True)
    if not args.medbin:
        print(vista_previa(records), flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
