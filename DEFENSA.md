# Guion de defensa: clasificacion de residuos hospitalarios

## Que resuelve

El sistema asigna una de siete clases a la imagen de un objeto: jeringas, guantes, mascarillas, gasas/vendas, frascos/ampollas, medicamentos y papel/carton. La app calcula la prediccion con un modelo entrenado y guardado localmente. La imagen de la webcam se procesa en el equipo, sin enviarse a servicios externos.

La alerta roja o verde se obtiene de una regla asociada al tipo de objeto. No se presenta como prueba de infeccion, caducidad o composicion quimica. La categoria de medicamentos contiene envases y presentaciones visuales; la camara no verifica la fecha de vencimiento.

## Relacion con las diapositivas

| Componente | Justificacion | Referencia |
|---|---|---|
| ReLU en capas ocultas | No linealidad; gradiente 1 en la zona positiva, con menor saturacion que sigmoid | 424 y 748 |
| Softmax de siete salidas | Clases exclusivas; probabilidades que suman 1 | 421, 424 y 749 |
| Cross-Entropy | Penaliza probabilidades bajas para la clase verdadera | 485 |
| Adam | Combina memoria del gradiente y adaptacion del paso | 610 |
| Train, validation y test | Aprender parametros, elegir configuracion y medir generalizacion final | 677 |
| F1 y AUC | Accuracy sola puede ocultar errores ante desbalance | 703; se adapta aqui a multiclase |
| Dropout y EarlyStopping | Control de alta varianza y generalizacion | 719 |
| Conv2D con same | Aprende filtros sin reducir dimensiones espaciales con stride 1 | 741 |
| MaxPooling2D | Reduce dimensiones y aporta invariancia local a pequenos desplazamientos | 742 |
| OpenCV, Tkinter, PIL, Keras | Familias de librerias visibles en el ejemplo de lentes | 750, pagina 50 del segundo PDF |

El PDF no contiene todo el archivo Con_Sin_Lentes_Pedagogico.py. La implementacion es propia y sigue la teoria visible y los requisitos proporcionados por el estudiante.

## Recorrido del codigo

1. `adquirir_datos.py`: obtiene imagenes publicas individuales y guarda su procedencia.
2. `preparar_datos.py`: convierte a RGB, conserva proporcion, ajusta a 128x128, controla duplicados y divide por grupos.
3. `Residuos_Medicos_Pedagogico.py`: contiene los imports y la CNN explicita.
4. `entrenamiento.py`: entrena con Adam/Cross-Entropy; monitorea val_loss y guarda resultados.
5. `interfaz.py`: captura webcam con OpenCV y dibuja la interfaz con Tkinter e ImageTk.

## Que explicar de la red

Entrada RGB 128x128x3 -> Rescaling(1/255) -> aumentos solo al entrenar -> Conv2D(16)+MaxPooling -> Conv2D(32)+MaxPooling -> Conv2D(64)+MaxPooling -> Flatten -> Dense(64, ReLU) -> Dropout(0.40) -> Dense(7, Softmax).

Los kernels de la CNN se aprenden por backpropagation. Los filtros no son reglas manuales que asignan nombres de objetos. `padding='same'` conserva las dimensiones de cada convolucion con stride 1; las capas de pooling se encargan de reducirlas. ReLU ayuda al flujo del gradiente positivo, aunque no garantiza por si sola que desaparezcan todos los problemas de gradiente.

Dropout actua durante entrenamiento. Los aumentos tambien se desactivan en inferencia. La normalizacion esta dentro del modelo, de modo que entrenamiento y webcam entregan pixeles 0..255 y no se normaliza dos veces.

## Como leer los resultados

- Train loss mide el ajuste a los ejemplos usados para aprender.
- Val loss estima el comportamiento en datos no usados para ajustar los pesos; EarlyStopping monitorea esta variable y recupera un buen punto del entrenamiento.
- Si train loss baja y val loss sube de forma sostenida, hay indicios de sobreajuste/alta varianza.
- Si ambos errores siguen altos, puede haber subajuste, problemas de datos o entrenamiento insuficiente.
- F1 macro promedia el F1 de cada clase con el mismo peso. Una clase dificil no queda oculta por una clase frecuente.
- F1 weighted pondera por soporte; se reporta como complemento.
- AUC macro OVR calcula una curva ROC para cada clase frente al resto y promedia sus areas. No se usa directamente la formula binaria sobre las siete etiquetas numericas.
- La matriz de confusion muestra que clases se confunden. Revisar especialmente jeringas y frascos/ampollas.

Train loss utiliza pesos de clase y se calcula con aumentos y Dropout activos; val_loss no emplea esos aumentos ni Dropout. Por ello no se espera que ambas curvas sean identicas, y una perdida de validacion menor al inicio no es por si sola un problema. La brecha se interpreta junto con la tendencia y las metricas por clase.

Se reportan los resultados obtenidos, incluidos errores. Accuracy o AUC altas sobre fotos publicas no garantizan igual rendimiento con cualquier webcam, fondo o iluminacion.

## Preparacion de la demostracion

1. Abrir `iniciar.bat` y esperar a que cargue el modelo.
2. Mostrar un solo objeto, centrado en el recuadro, cercano y con buena luz.
3. Explicar la distribucion de las siete probabilidades y su suma.
4. Mostrar una alerta roja y una verde con ejemplos adecuados.
5. Abrir Ver metricas: entrenamiento, matriz de confusion y curvas ROC.
6. Usar Elegir imagen o Ejemplo de prueba si la camara falla; estos botones tambien usan el modelo real.

El umbral de aceptacion y la diferencia entre las dos clases mas probables permiten mostrar No concluyente. Este mecanismo reduce decisiones de baja confianza, pero una red de conjunto cerrado puede asignar alta probabilidad a objetos desconocidos. No se afirma deteccion universal de objetos fuera del dataset.

## Preguntas frecuentes del docente

**Como demuestro que la red esta implementada en este proyecto?** Teachable Machine tambien usa redes neuronales. La arquitectura y el entrenamiento original se pueden mostrar en construir_modelo de Residuos_Medicos_Pedagogico.py y modelo.fit de entrenamiento.py. El resumen muestra 1072679 parametros entrenables; la primera convolucion tiene 3*3*3*16+16 = 448 parametros. Para la demostracion actual, la app usa una CNN ajustada con 83 fotos propias y compara sus rasgos con esas fotos. El informe y las metricas de las diapositivas corresponden al checkpoint original, anterior al ajuste.

Para una prueba breve ejecuta demostrar_red.bat. Construye una red nueva en memoria, compara su arquitectura con la del modelo activo y comprueba que el archivo activo coincide con el checkpoint local. Usa 14 ejemplos de entrenamiento para tres pasos explicitos con GradientTape y Adam, imprime gradientes y cambios de pesos, y guarda salidas/demostracion_red.json. El modelo de la aplicacion queda intacto. Esta prueba muestra el mecanismo de aprendizaje, no mide calidad de generalizacion. La perdida puede oscilar entre los tres pasos por los aumentos, Dropout y la actualizacion estocastica.

La evidencia practica del entrenamiento es poder leer el codigo, explicar sus decisiones y volver a entrenar desde la inicializacion. En el modo actual, la similitud con fotos propias es un apoyo adicional para reconocer esos mismos objetos; no es una nueva medicion independiente de generalizacion.

**Por que no use el modelo YOLO encontrado?** Porque la arquitectura solicitada es una CNN Sequential explicita. Reutilizo datos y anotaciones; el modelo final sigue el planteamiento de la materia.

**Por que separar por grupos?** Las variantes de una misma foto no deben distribuirse entre train, validation y test. Se agrupan nombres de origen y similitudes perceptuales antes de dividir. Esta heuristica reduce fugas evidentes, pero no demuestra independencia de todos los objetos fisicos fotografiados.

**Que significa un 90%?** Es una probabilidad estimada por el modelo entre sus clases entrenadas; no equivale a una garantia clinica o de descontaminacion.

**Por que hay siete neuronas al final?** Porque son siete clases exclusivas y la salida softmax produce una distribucion sobre ellas.

**Por que hay hilos en la interfaz?** Para mantener la ventana responsive mientras se capturan y procesan cuadros. Solo el hilo principal actualiza Tkinter.
