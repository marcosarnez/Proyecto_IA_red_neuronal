# Clasificación de residuos hospitalarios con una CNN

Proyecto académico de Marcos Arnez Pocomani. La aplicación de escritorio usa Python, TensorFlow/Keras, OpenCV y Tkinter para clasificar una imagen en siete categorías visuales: jeringas, guantes, mascarillas, gasas y vendas, frascos y ampollas, medicamentos, y papel y cartón.

## Abrir la aplicación en Windows

1. Instala **Python 3.12** con el lanzador `py`.
2. Ejecuta `instalar_dependencias.bat` una vez. Descarga las bibliotecas de `requirements.txt` y crea `.venv`.
3. Ejecuta `iniciar.bat`. Mantén abierta la consola durante la carga del modelo.

La aplicación abre la webcam. Centra un solo objeto dentro del recuadro y usa buena luz. «Elegir imagen» permite cargar una foto local. El botón «Ejemplo de prueba» necesita las imágenes preparadas, que no se distribuyen en este repositorio.

## Red neuronal y resultados

La arquitectura está definida en `Residuos_Medicos_Pedagogico.py`. Es una CNN `Sequential` con tres bloques Conv2D y MaxPooling, una capa densa y siete salidas Softmax. `entrenamiento.py` contiene el entrenamiento con Adam y entropía cruzada. El checkpoint inicial está en `salidas/modelo_residuos.keras`.

El checkpoint inicial acertó 158 de 210 imágenes de prueba (75,24 %) y obtuvo F1 macro de 0,750. Estos resultados aparecen en el informe y en la presentación. Después se ajustó otra versión de la CNN con 83 fotografías propias de los objetos de la demostración. La aplicación carga esa versión desde `salidas/ajuste_fotos_propias/modelo_candidato.keras` y compara sus rasgos con `referencias.npz`. Cuando muestra «similitud», el porcentaje expresa parecido con esas referencias. Las cifras de la presentación corresponden al checkpoint inicial.

Las 83 fotos sirvieron para adaptar la demostración. Comprobar el sistema con esas mismas fotos no mide su desempeño en objetos nuevos. `salidas/ajuste_fotos_propias/resultado.json` registra el ajuste y su límite de evaluación.

## Archivos del proyecto

- `iniciar.bat`, `interfaz.py` y `lanzar.pyw` abren la aplicación.
- `Residuos_Medicos_Pedagogico.py` define la CNN y `entrenamiento.py` la entrena.
- `preparar_datos.py` prepara imágenes y `adquirir_datos.py` documenta una obtención acotada de fuentes públicas.
- `ajustar_fotos_propias.py` y `preparar_referencias_defensa.py` reproducen el ajuste cuando se dispone de las fotos propias y de su archivo de etiquetas.
- `salidas/` contiene los modelos, las etiquetas y las métricas.
- `defensa/` contiene la presentación y el guion.
- `FUENTES.md` documenta la procedencia de las imágenes usadas en la etapa inicial.

Las fotografías de los objetos y las imágenes originales/preparadas del entrenamiento se mantienen fuera de este repositorio. `datos/manifiesto.json` conserva los metadatos de preparación, pero sus rutas de imagen requieren recuperar los archivos originales para volver a entrenar. El modelo guardado permite ejecutar la aplicación sin esas fotos.

## Informe

`Informe_Revista_3i_Residuos_Hospitalarios_Marcos_Arnez_Pocomani.docx` recoge el método, resultados y limitaciones del proyecto. La clasificación de peligro que muestra la interfaz es una orientación académica por tipo visual de objeto. No identifica contaminación ni el contenido real de un envase.
