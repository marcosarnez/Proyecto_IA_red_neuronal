# Guion final de defensa — Marcos Arnez Pocomani

## La idea en veinte segundos

«Construí una red neuronal convolucional para reconocer siete tipos de residuos hospitalarios. La entrené localmente en Python. Al probar mis objetos reales, tomé 83 fotos adicionales y ajusté la red. Para la demostración, también comparo los rasgos que extrae la CNN con esas fotos propias. Así puedo mostrar el funcionamiento con cámara y explicar de dónde sale cada resultado».

## Diapositiva por diapositiva

1. **Portada.** «Mi proyecto reconoce objetos hospitalarios desde una cámara y muestra una orientación por categoría».
2. **Red neuronal.** «Usé una CNN. Su trabajo es aprender detalles de las imágenes, como bordes, texturas y formas».
3. **Siete clases.** «La red entrega una puntuación para cada clase. Hay una salida para jeringas, guantes, barbijos, gasas, frascos, medicamentos y papel o cartón».
4. **Neurona.** «Cada neurona combina números con pesos. Entrenar significa ajustar esos pesos según los errores que comete la red».
5. **Convolución.** «Un mismo filtro busca un detalle en distintas partes de la imagen. Por eso una forma puede reconocerse aunque se mueva un poco».
6. **Elección del algoritmo.** «Elegí CNN porque clasifico un objeto centrado. YOLO serviría si necesitara localizar varios objetos en una escena. Al final agregué una comparación con mis fotos, usando los rasgos de la CNN para adaptar la demostración a objetos reales».
7. **Arquitectura.** «La foto entra con un tamaño fijo. Pasa por tres bloques que extraen detalles y termina en siete salidas. Todo está definido en mi código».
8. **Tres operaciones.** «Convolución busca detalles; ReLU ayuda a que la red aprenda relaciones complejas; MaxPooling reduce la información conservando los rasgos más destacados».
9. **Aprendizaje.** «La red predice, compara con la respuesta correcta, calcula el error y modifica sus pesos. Repetí ese ciclo durante el entrenamiento».
10. **Adam.** «Adam es el método que usé para ajustar los pesos. También controlé cuándo detener el aprendizaje para evitar que la red se acostumbre demasiado a las imágenes de entrenamiento».
11. **Evidencia de entrenamiento.** «Puedo abrir el código de las capas y el entrenamiento. Esta diapositiva muestra el checkpoint original. Después guardé por separado una versión ajustada con mis 83 fotos».
12. **Selección del modelo inicial.** «Durante el entrenamiento inicial elegí la versión con menor error de validación, en la época 22».
13. **Resultado inicial.** «Antes del ajuste con mis objetos, ese modelo acertó 158 de 210 imágenes de prueba, aproximadamente el 75 %. Es el resultado que aparece en el informe y en estas diapositivas».
14. **Errores iniciales.** «La tabla muestra qué objetos se confundieron. Por ejemplo, hubo medicamentos que se parecían visualmente a frascos».
15. **Aplicación.** «La cámara captura una imagen, la CNN calcula rasgos y la aplicación muestra la clase. En la versión de demostración también compara esos rasgos con mis fotos. Cuando dice *similitud*, ese porcentaje indica parecido con las referencias».
16. **Archivos y CRUD.** «El entrenamiento crea el modelo y los resultados. La aplicación los lee. Al mejorar el entrenamiento se puede actualizar el modelo. Un historial editable con opción de borrar sería una ampliación».
17. **Demostración.** «Voy a mostrar mis objetos de uno en uno, centrados en el recuadro. También tengo siete imágenes listas por si cambia la luz o falla la cámara».
18. **Conclusión.** «Logré construir y entrenar la CNN, integrarla con webcam y ajustarla para mis objetos. La siguiente prueba sería fotografiar objetos nuevos en otros fondos y medir el resultado».
19. **Anexo de clases.** «El programa funciona mejor en unas clases que en otras. Revisar cada clase evita esconder los errores detrás de un solo promedio».
20. **Relación con la materia.** «Apliqué redes neuronales, activaciones, cálculo de error, optimización, convolución y evaluación. El ejemplo pedagógico de lentes me sirvió para conectar la red con la interfaz».
21. **Pregunta principal.** «El algoritmo central es una CNN supervisada. La entrené ajustando sus pesos con Adam. En la demo añadí comparación de rasgos con mis fotos para reconocer mejor esos mismos objetos».

## Explicación sencilla de la CNN si el docente se detiene en la diapositiva 7

«Imagínese que la red mira una foto por partes. Los primeros filtros se fijan en líneas o bordes. Los siguientes combinan esos detalles para formar patrones más grandes. Al final reúne lo que encontró y decide qué clase se parece más a la imagen. Yo escribí la estructura de esas capas y ejecuté el entrenamiento en mi computadora».

## Demostración después de las diapositivas

1. Abre `iniciar.bat` y deja la consola abierta durante la carga.
2. Enseña primero el guante, el barbijo o la gasa frente a la webcam. Coloca solo un objeto dentro del recuadro y busca buena luz.
3. Señala la clase y el texto de debajo. Si dice «Similitud con fotos propias», explica que la CNN encontró rasgos parecidos a tus fotografías de referencia.
4. Enseña la jeringa nueva sin aguja. Luego enseña un frasco. El frasco plástico de alcohol se muestra como envase que requiere revisar el contenido, no como vidrio.
5. Muestra el blíster. Para la séptima clase, usa un trozo de cartón o la imagen `07_Papel_y_carton.png`.
6. Si la webcam falla con un objeto, pulsa «Elegir imagen» y abre los archivos de `defensa/imagenes_para_demo`, numerados del 01 al 07. Las siete imágenes fueron comprobadas en la aplicación.
7. «Ver métricas» muestra la evaluación del modelo inicial, anterior al ajuste rápido con las fotos propias.

## Respuestas cortas

**¿Qué algoritmo usaste?** «Una CNN de clasificación de imágenes. La entrené con Keras y Adam. En la demostración comparo sus rasgos con fotos propias».

**¿Por qué CNN?** «Porque aprovecha los detalles cercanos entre píxeles y reutiliza sus filtros en distintas zonas de la imagen».

**¿Es una red tuya?** «Sí. Puedo mostrar el archivo donde definí sus capas y el código que ejecutó el entrenamiento. El ajuste final usa mis fotos de los objetos físicos».

**¿Qué significan los números del informe?** «Corresponden al modelo inicial, evaluado antes del ajuste. Las 83 fotos propias sirvieron para adaptar la demostración y no son una evaluación independiente».

**¿De dónde salieron las imágenes?** «El entrenamiento inicial usó imágenes públicas que preparé y organicé. Después tomé estas 83 fotos de mis objetos y ajusté la CNN para la demostración. El código de entrenamiento y los archivos resultantes están en el proyecto».

**¿Por qué una imagen puede fallar en vivo?** «La iluminación, el fondo y la posición cambian. La aplicación muestra *No concluyente* en algunos casos; por eso probé mis objetos y guardé imágenes de respaldo».
