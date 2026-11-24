# Procedencia y atribucion

## Biomedical Waste Classification Images Dataset

- Autor/distribuidor: mario78, Kaggle.
- https://www.kaggle.com/datasets/mario78/biomedical-waste-classification-images-dataset
- Licencia publicada en la ficha: MIT.
- Se seleccionaron 160 imagenes por cada una de seis familias, usando archivos individuales de Dataset/Raw. No se descargo el archivo completo de 5.5 GB.
- Carpetas: used_syringes, used_medical_gloves, used_masks, blood_soaked_bandages, ampoules_full, ampuoles_broken, vaccine_or_medicine_vials y expired_tablets.
- Existen variantes geometricas y fotografias de objetos similares; la separacion incluye agrupacion por nombres y hashes perceptuales. Esto no garantiza independencia de todos los objetos fisicos de origen.
- La etiqueta expired_tablets se usa como categoria visual Medicamentos/blisteres. No se entrena ni se afirma deteccion de caducidad.
- La ficha y el indice consultados se conservan en datos/metadatos/kaggle_ficha.json y kaggle_archivos.json. Cada imagen seleccionada tiene ruta y enlace de procedencia en plan_kaggle.json.

## MedBin / MedBinNet

- https://github.com/Wayne3918/MedbinNet
- Commit: fee9be8d9aeea4d9ea05384be648ea8b178d5262.
- Datos: CC BY 4.0, segun las anotaciones COCO. Codigo original: Apache 2.0. Este proyecto no utiliza sus pesos ni su implementacion MMDetection.
- Se seleccionaron hasta 60 imagenes de origen por familia medica y se recortaron objetos a partir de sus cuadros anotados. Los recortes de un mismo origen mantienen un unico grupo.
- Referencia: Xu, X. et al. (2025). MedBin: A lightweight End-to-End model-based method for medical waste management. Waste Management, 200, 114742. https://doi.org/10.1016/j.wasman.2025.114742
- El articulo describe otra version de los datos. Las clases efectivamente usadas se obtienen de los JSON conservados en datos/metadatos/medbin_train.json y medbin_valid.json.
- Hay una exclusion manual documentada de un recorte ambiguo. La auditoria no equivale a una validacion clinica completa de las etiquetas.

## TrashNet

- Autores: Gary Thung y Mindy Yang.
- https://github.com/garythung/trashnet
- Licencia MIT publicada en el repositorio original.
- Se extrajeron 160 imagenes de papel y carton desde dataset-resized.zip (42.8 MB). Las demas clases del ZIP no se incluyen en el manifiesto de entrenamiento.

## Material de la materia

- D:/Descargas/slide 415- 700 Inteligencia Artificial.pdf
- D:/Descargas/Slide 701 - 750 Inteligencia Artificial.pdf
- Las referencias de cada componente se detallan en DEFENSA.md. La pagina 50 del segundo PDF muestra una captura parcial de Con_Sin_Lentes_Pedagogico.py, no su codigo completo.

## Uso local

Las imagenes de webcam se procesan en memoria. La aplicacion de demostracion no las guarda ni las transmite a terceros. Las descargas preparatorias solo solicitan datos publicos de las fuentes anteriores.
