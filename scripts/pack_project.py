"""
pack_project.py — Script para empaquetar el proyecto para Colab.

Crea un archivo ZIP con todo lo necesario para entrenar en la nube,
ignorando archivos pesados (checkpoints, venv, git, etc.).
"""

import zipfile
from pathlib import Path


def pack_project():
    source_dir = Path(".")
    output_zip = source_dir / "nano-language-model.zip"

    # Archivos/carpetas a ignorar
    ignore_list = {
        ".venv",
        "__pycache__",
        ".git",
        ".vscode",
        ".idea",
        "archive",
        "demo",
        "models/checkpoints",
        "logs",
        "nano-language-model.zip",
        "test_results",
        "wandb",
    }

    # Extensiones a ignorar
    ignore_ext = {".pyc", ".pyo", ".pyd"}

    print(f"Empaquetando proyecto en {output_zip}...")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            # Excluir rutas absolutas que contengan partes ignoradas
            parts = file_path.parts
            if any(p in ignore_list for p in parts):
                continue

            if file_path.suffix in ignore_ext:
                continue

            if file_path.is_dir():
                continue

            # Calcular ruta relativa dentro del zip
            arcname = file_path.relative_to(source_dir)

            print(f"  + {arcname}")
            zf.write(file_path, arcname)

    print(f"\nListo! Sube 'nano-language-model.zip' a tu Google Drive.")
    print("   Tamaño: {:.2f} MB".format(output_zip.stat().st_size / (1024 * 1024)))


if __name__ == "__main__":
    pack_project()
