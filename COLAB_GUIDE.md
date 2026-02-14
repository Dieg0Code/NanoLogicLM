# Guía de Despliegue en Google Colab 🚀

Si nunca has usado Colab, no te preocupes. Es básicamente una computadora prestada de Google que puedes usar desde tu navegador.

## Paso 1: Subir archivos a Google Drive

1.  Ve a [Google Drive](https://drive.google.com).
2.  Crea una carpeta llamada `AI` (o úsala en la raíz si prefieres).
3.  Arrastra y suelta el archivo **`nano-language-model.zip`** (que generamos con `scripts/pack_project.py`) dentro de esa carpeta.
    *   Ruta recomendada: `Mi unidad > AI > nano-language-model.zip`
4.  Sube también el archivo **`notebooks/Start_Training_Colab.ipynb`** a la misma carpeta.

## Paso 2: Abrir el Notebook en Colab

1.  Dale doble clic al archivo `Start_Training_Colab.ipynb` en Google Drive.
2.  Se abrirá una nueva pestaña con la interfaz de Colab.
    *   Si no se abre automáticamente, ve a [Google Colab](https://colab.research.google.com), dale a "File > Upload notebook" y sube el archivo `.ipynb`.

## Paso 3: Configurar la GPU (Importante)

1.  En el menú de arriba, ve a **Runtime** (o Entorno de ejecución) > **Change runtime type** (Cambiar tipo de entorno).
2.  En "Hardware accelerator", selecciona **T4 GPU**.
3.  Dale a **Save**.
    *   Esto te conecta a una tarjeta gráfica real. Si no lo haces, el entrenamiento será eterno (CPU).

## Paso 4: Ejecutar el Notebook

1.  Verás varias "celdas" de código (bloques grises con texto).
2.  Haz clic en el botón de **Play (▶️)** que aparece a la izquierda de cada celda.
    *   **Celda 1 (Mount Drive)**: Te pedirá permiso para acceder a tu Google Drive. Dale "Connect to Google Drive" y acepta. Esto es necesario para leer el `.zip`.
    *   **Celda 2 (Unzip)**: Descomprime el proyecto. Verás una lista de archivos.
    *   **Celda 3 (Install)**: Instala las librerías necesarias.
    *   **Celda 6 (Train)**: ¡Aquí empieza la magia! Verás la barra de progreso del entrenamiento.

## Paso 5: Guardar tus Checkpoints

*   El notebook está configurado para copiar automáticamente los checkpoints (los archivos del modelo entrenado) de vuelta a tu Google Drive (`Mi unidad > nano-checkpoints`).
*   Así, si Colab se desconecta, no pierdes tu progreso.

---

### ¿Problemas comunes?

*   **"File not found"**: Revisa que el `.zip` esté exactamente en la ruta que dice el código (`/content/drive/MyDrive/...`). Si lo pusiste en una carpeta, ajusta la ruta en la Celda 2.
*   **"CUDA out of memory"**: Si te pasa esto, reduce el `batch_size` en el comando de entrenamiento (ej. cambia 32 por 16).
