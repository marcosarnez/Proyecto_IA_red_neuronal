"""Interfaz Tkinter; captura e inferencia en hilos, actualizaciones Tk en el principal."""
import json
import hashlib
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf

from configuracion import RAIZ, CLASES, NOMBRES, PELIGRO, COLORES, MODELO, MAPA, TAMANO_IMAGEN, MODELO_DEMO, REFERENCIAS_DEMO


def preparar_entrada(image):
    # El modelo contiene Rescaling(1./255): aqui se conservan pixeles 0..255.
    image = ImageOps.pad(image.convert("RGB"), (TAMANO_IMAGEN, TAMANO_IMAGEN), method=Image.Resampling.BILINEAR, color=(127, 127, 127))
    return np.asarray(image, dtype=np.float32)[None]


class Aplicacion:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.root.title("Clasificación de residuos hospitalarios | CNN pedagógica")
        self.root.geometry("1120x760")
        self.root.minsize(1000, 690)
        self.root.configure(bg="#eef2f6")
        self.stop = threading.Event()
        self.camera_stop = threading.Event()
        self.frames = queue.Queue(maxsize=1)
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=3)
        self.cap = None
        self.camera_thread = None
        self.mode = "espera"
        self.version = 0
        self.loaded = False
        self.last_request = 0.0
        self.ema = None
        self.ema_type = None
        self.qa = {"modelo_cargado": False, "camara_abierta": False, "frames": 0,
                   "inferencias": 0, "referencias": 0, "ultimo_resultado": None, "errores": []}
        self.threshold = tk.DoubleVar(value=0.65)
        self.camera_index = tk.IntVar(value=args.camara)
        self._construir()
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)
        threading.Thread(target=self._inferencia, daemon=True, name="InferenciaCNN").start()
        self.root.after(50, self._actualizar)
        if args.autocerrar:
            self.root.after(args.autocerrar*1000, self.cerrar)

    def _construir(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 9))
        style.configure("TProgressbar", thickness=11, troughcolor="#e2e8f0", background="#2979bd")
        header = tk.Frame(self.root, bg="#15304d", padx=24, pady=18)
        header.pack(fill="x")
        tk.Label(header, text="RESIDUOS HOSPITALARIOS", font=("Segoe UI", 21, "bold"), fg="white", bg="#15304d").pack(anchor="w")
        tk.Label(header, text="CNN entrenada localmente · Ajuste con fotos propias y comparación visual", font=("Segoe UI", 10), fg="#c1d3e4", bg="#15304d").pack(anchor="w")
        toolbar = tk.Frame(self.root, bg="#eef2f6", padx=20, pady=13)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Abrir cámara", command=self.abrir_camara).pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Elegir imagen", command=self.elegir_imagen).pack(side="left", padx=8)
        ttk.Button(toolbar, text="Ejemplo de prueba", command=self.ejemplo).pack(side="left", padx=8)
        ttk.Button(toolbar, text="Ver métricas", command=self.ver_metricas).pack(side="left", padx=8)
        tk.Label(toolbar, text="Cámara:", bg="#eef2f6").pack(side="left", padx=(20, 3))
        ttk.Spinbox(toolbar, from_=0, to=5, width=3, textvariable=self.camera_index).pack(side="left")
        content = tk.Frame(self.root, bg="#eef2f6", padx=20)
        content.pack(fill="both", expand=True)
        left = tk.Frame(content, bg="white", padx=12, pady=12)
        left.pack(side="left", fill="both", expand=True)
        self.video = tk.Label(left, bg="#152232", fg="#c7d2df", text="Cargando el modelo local…", font=("Segoe UI", 14), width=62)
        self.video.pack(fill="both", expand=True)
        self.mode_label = tk.Label(left, text="Coloca un solo objeto dentro del recuadro y ocupa buena parte de él.", bg="white", fg="#53677c", font=("Segoe UI", 10), wraplength=640, pady=10)
        self.mode_label.pack(fill="x")
        right = tk.Frame(content, bg="white", padx=18, pady=16, width=330)
        right.pack(side="right", fill="y", padx=(15, 0))
        right.pack_propagate(False)
        tk.Label(right, text="RESULTADO", bg="white", fg="#61768c", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.class_label = tk.Label(right, text="Preparando…", bg="white", fg="#15304d", font=("Segoe UI", 19, "bold"), wraplength=285, justify="left")
        self.class_label.pack(anchor="w", pady=(8, 3))
        self.prob_label = tk.Label(right, text="", bg="white", fg="#53677c", font=("Segoe UI", 12))
        self.prob_label.pack(anchor="w")
        self.alert = tk.Label(right, text="ESPERANDO IMAGEN", bg="#edf1f5", fg="#53677c", font=("Segoe UI", 10, "bold"), wraplength=270, padx=10, pady=14)
        self.alert.pack(fill="x", pady=16)
        self.distribution_label = tk.Label(right, text="Distribución de probabilidades CNN", bg="white", fg="#15304d", font=("Segoe UI", 10, "bold"))
        self.distribution_label.pack(anchor="w", pady=(2, 8))
        self.bars, self.percentages = [], []
        for name in NOMBRES:
            line = tk.Frame(right, bg="white")
            line.pack(fill="x", pady=(4, 1))
            tk.Label(line, text=name, bg="white", fg="#40556c", font=("Segoe UI", 9)).pack(side="left")
            percent = tk.Label(line, text="—", bg="white", fg="#40556c", font=("Segoe UI", 9))
            percent.pack(side="right")
            bar = ttk.Progressbar(right, maximum=100)
            bar.pack(fill="x")
            self.bars.append(bar)
            self.percentages.append(percent)
        tk.Label(right, text="Umbral de aceptación CNN", bg="white", fg="#53677c", font=("Segoe UI", 9)).pack(anchor="w", pady=(16, 0))
        tk.Scale(right, from_=0.45, to=0.95, resolution=0.05, orient="horizontal", variable=self.threshold, bg="white", highlightthickness=0).pack(fill="x")
        self.status = tk.Label(self.root, text="Iniciando…", anchor="w", bg="#eef2f6", fg="#53677c", padx=22, pady=9, font=("Segoe UI", 9))
        self.status.pack(fill="x")
        tk.Label(self.root, text="Orientación académica por tipo de objeto. La imagen no confirma contaminación ni caducidad.", bg="#eef2f6", fg="#61768c", font=("Segoe UI", 9), pady=5).pack(fill="x")

    @staticmethod
    def _poner(q, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(item)

    def _inferencia(self):
        try:
            metadata = json.loads(MAPA.read_text(encoding="utf-8"))
            if metadata["clases"] != CLASES or metadata["tamano_imagen"] != TAMANO_IMAGEN:
                raise ValueError("Modelo y etiquetas no coinciden")
            calibrated = MODELO_DEMO.is_file() and REFERENCIAS_DEMO.is_file()
            model_path = MODELO_DEMO if calibrated else MODELO
            model = tf.keras.models.load_model(model_path, compile=False)
            model(np.zeros((1, TAMANO_IMAGEN, TAMANO_IMAGEN, 3), dtype=np.float32), training=False)
            if calibrated:
                reference_path = REFERENCIAS_DEMO.with_suffix(".json")
                reference_meta = json.loads(reference_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
                if reference_meta["clases"] != CLASES or reference_meta["modelo_sha256"] != digest:
                    raise ValueError("Las fotos de referencia no corresponden a esta CNN")
                with np.load(REFERENCIAS_DEMO) as data:
                    features = data["features"]
                    labels = data["labels"]
                analyzer = tf.keras.Model(model.inputs, [model.outputs[0], model.get_layer("clasificador").output])
            self._poner(self.results, ("listo", metadata))
            while not self.stop.is_set():
                try:
                    image, version = self.requests.get(timeout=0.2)
                except queue.Empty:
                    continue
                start = time.monotonic()
                input_image = preparar_entrada(image)
                if calibrated:
                    raw_scores, embedding = analyzer([input_image], training=False)
                    p = np.asarray(raw_scores)[0]
                    embedding = np.asarray(embedding)[0]
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                    similarity = features @ embedding
                    class_similarity = np.zeros(len(CLASES), dtype=np.float32)
                    for class_id in np.unique(labels):
                        class_similarity[class_id] = np.max(similarity[labels == class_id])
                    nearest = int(class_similarity.argmax())
                    if class_similarity[nearest] >= 0.94 and p[-1] < 0.5:
                        self._poner(self.results, ("referencia", class_similarity, nearest,
                                                  version, (time.monotonic() - start) * 1000))
                        continue
                else:
                    p = np.asarray(model(input_image, training=False))[0]
                if not np.isfinite(p).all() or not np.isclose(p.sum(), 1, atol=1e-5):
                    raise ValueError("Probabilidades invalidas")
                self._poner(self.results, ("prediccion", p, version, (time.monotonic()-start)*1000))
        except Exception as error:
            self._poner(self.results, ("error", str(error)))

    def abrir_camara(self):
        if not self.loaded:
            self.status.configure(text="Espera a que termine de cargar el modelo.")
            return
        self.camera_stop.set()
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1)
        self.camera_stop = threading.Event()
        self.version += 1
        self.mode, self.ema, self.ema_type = "camara", None, None
        index, stop_event = self.camera_index.get(), self.camera_stop

        def capture():
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if not cap.isOpened():
                self._poner(self.results, ("error_camara", "No se pudo abrir la cámara. Revisa el índice, permisos y otras aplicaciones."))
                cap.release()
                return
            self.cap = cap
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.qa["camara_abierta"] = True
            try:
                while not stop_event.is_set() and not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    self._poner(self.frames, frame)
                    time.sleep(0.02)
            finally:
                cap.release()

        self.camera_thread = threading.Thread(target=capture, daemon=True, name="Webcam")
        self.camera_thread.start()
        self.mode_label.configure(text="Cámara en vivo: un objeto centrado, cercano y con buena iluminación.")

    def _mostrar(self, image):
        width, height = max(500, self.video.winfo_width()), max(350, self.video.winfo_height())
        photo = ImageOps.contain(image, (width, height))
        self.tk_photo = ImageTk.PhotoImage(photo)
        self.video.configure(image=self.tk_photo, text="")

    def cargar_archivo(self, path):
        self.camera_stop.set()
        self.mode, self.ema, self.ema_type = "imagen", None, None
        self.version += 1
        try:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            self._mostrar(image)
            self.mode_label.configure(text="Archivo: " + str(path).replace("\\", "/").split("/")[-1])
            self._poner(self.requests, (image.copy(), self.version))
        except OSError as error:
            messagebox.showerror("Imagen", str(error))

    def elegir_imagen(self):
        if not self.loaded:
            return
        path = filedialog.askopenfilename(title="Elegir imagen de residuo", filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            self.cargar_archivo(path)

    def ejemplo(self):
        if not self.loaded:
            return
        manifest = json.loads((RAIZ / "datos" / "manifiesto.json").read_text(encoding="utf-8"))
        examples = [r for r in manifest["imagenes"] if r["particion"] == "test"]
        self.example_index = (getattr(self, "example_index", -1) + 1) % len(CLASES)
        row = next(r for r in examples if r["clase_id"] == self.example_index)
        self.cargar_archivo(RAIZ / row["ruta"])
        self.mode_label.configure(text="Ejemplo reservado de test · Etiqueta real: " + NOMBRES[row["clase_id"]])

    def _resultado(self, p, milliseconds):
        self.qa["inferencias"] += 1
        if self.mode == "camara":
            if self.ema_type != "cnn":
                self.ema = None
            self.ema_type = "cnn"
            self.ema = p if self.ema is None else 0.55*p + 0.45*self.ema
            p = self.ema
        index = int(p.argmax())
        ordered = np.sort(p)
        accepted = p[index] >= self.threshold.get() and ordered[-1]-ordered[-2] >= 0.12
        self.class_label.configure(text=NOMBRES[index] if accepted else "No concluyente")
        self.qa["ultimo_resultado"] = NOMBRES[index] if accepted else "No concluyente"
        self.distribution_label.configure(text="Distribución de probabilidades CNN")
        self.prob_label.configure(text=f"{p[index]*100:.1f}% · {NOMBRES[index]}" if not accepted else f"Probabilidad del modelo: {p[index]*100:.1f}%")
        if accepted:
            self.alert.configure(text=PELIGRO[index], bg=COLORES[index], fg="white")
        else:
            self.alert.configure(text="ACERCA Y CENTRA EL OBJETO", bg="#fff0cf", fg="#815b06")
        for bar, label, value in zip(self.bars, self.percentages, p):
            bar["value"] = float(value)*100
            label.configure(text=f"{value*100:.1f}%")
        self.status.configure(text=f"Inferencia local: {milliseconds:.0f} ms · Suma de probabilidades: {p.sum():.3f}")

    def _resultado_referencia(self, similarities, index, milliseconds):
        self.qa["inferencias"] += 1
        self.qa["referencias"] += 1
        if self.mode == "camara":
            if self.ema_type != "referencia":
                self.ema = None
            self.ema_type = "referencia"
            self.ema = similarities if self.ema is None else 0.55 * similarities + 0.45 * self.ema
            similarities = self.ema
            index = int(similarities.argmax())
        self.qa["ultimo_resultado"] = NOMBRES[index]
        self.class_label.configure(text=NOMBRES[index])
        self.prob_label.configure(text=f"Similitud con fotos propias: {similarities[index] * 100:.1f}%")
        self.alert.configure(text=PELIGRO[index], bg=COLORES[index], fg="white")
        self.distribution_label.configure(text="Similitud de rasgos de la CNN por clase")
        for bar, label, value in zip(self.bars, self.percentages, similarities):
            bar["value"] = float(value) * 100
            label.configure(text=f"{value * 100:.1f}%" if value > 0 else "—")
        self.status.configure(text=f"CNN ajustada y comparación con fotos propias · {milliseconds:.0f} ms")

    def _actualizar(self):
        if self.stop.is_set():
            return
        try:
            while True:
                result = self.results.get_nowait()
                if result[0] == "listo":
                    self.loaded = self.qa["modelo_cargado"] = True
                    self.status.configure(text="Modelo local cargado. Preparando cámara…")
                    if self.args.imagen:
                        self.cargar_archivo(self.args.imagen)
                    else:
                        self.abrir_camara()
                elif result[0] == "prediccion" and result[2] == self.version:
                    self._resultado(result[1], result[3])
                elif result[0] == "referencia" and result[3] == self.version:
                    self._resultado_referencia(result[1], result[2], result[4])
                elif result[0] in ("error", "error_camara"):
                    self.qa["errores"].append(result[1])
                    self.status.configure(text=result[1])
                    if result[0] == "error":
                        self.video.configure(text="No se pudo cargar el modelo.\nConsulta el mensaje inferior.")
        except queue.Empty:
            pass
        if self.mode == "camara":
            try:
                frame = self.frames.get_nowait()
                self.qa["frames"] += 1
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                size = int(min(h, w)*0.72)
                x, y = (w-size)//2, (h-size)//2
                roi = cv2.cvtColor(frame[y:y+size, x:x+size], cv2.COLOR_BGR2RGB)
                if time.monotonic()-self.last_request >= 0.20 and self.loaded:
                    self._poner(self.requests, (Image.fromarray(roi), self.version))
                    self.last_request = time.monotonic()
                cv2.rectangle(frame, (x, y), (x+size, y+size), (70, 210, 210), 2)
                self._mostrar(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            except queue.Empty:
                pass
        self.root.after(40, self._actualizar)

    def ver_metricas(self):
        window = tk.Toplevel(self.root)
        window.title("Resultados de la CNN inicial antes del ajuste con fotos propias")
        window.geometry("1050x630")
        tabs = ttk.Notebook(window)
        tabs.pack(fill="both", expand=True)
        for filename, label in (("graficos_entrenamiento.png", "Aprendizaje"), ("matriz_confusion.png", "Matriz de confusión"), ("curvas_roc.png", "AUC-ROC")):
            page = ttk.Frame(tabs)
            tabs.add(page, text=label)
            path = RAIZ / "salidas" / filename
            if path.exists():
                photo = ImageTk.PhotoImage(ImageOps.contain(Image.open(path), (1010, 560)))
                widget = tk.Label(page, image=photo)
                widget.image = photo
                widget.pack(expand=True)

    def cerrar(self):
        if self.stop.is_set():
            return
        self.stop.set()
        self.camera_stop.set()
        if self.args.autocerrar:
            (RAIZ / "salidas" / "prueba_interfaz.json").write_text(json.dumps(self.qa, indent=2), encoding="utf-8")
        self.root.destroy()


def ejecutar(args):
    root = tk.Tk()
    Aplicacion(root, args)
    root.mainloop()
