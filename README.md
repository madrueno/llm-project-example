# Proyecto de Ejemplo - Banking Intent Classification

**Repositorio didáctico que presenta buenas prácticas para proyectos de experimentación con modelos extensos de lenguaje**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-gestor%20de%20paquetes-5A67D8)](https://docs.astral.sh/uv/)
[![VSCode](https://img.shields.io/badge/VSCode-IDE-007ACC?logo=visualstudiocode&logoColor=white)](https://code.visualstudio.com/)
[![Ruff](https://img.shields.io/badge/estilo-ruff-black)](https://docs.astral.sh/ruff/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](LICENSE)

## Indice de contenidos

- [Descripción](#descripción)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
- [Licencia](#licencia)

## Descripción

Proyecto para clasificar intenciones bancarias basado en el dataset [Banking77](https://huggingface.co/datasets/mteb/banking77) mediante un pipeline reproducible de experimentación con modelos extensos de lenguaje.

Trabaja con un dataset derivado de *Banking77*, incluyendo diversos experimentos que comparan múltiples estrategias de clasificación de texto con modelos extensos de lenguaje. Mantiene una organización modular, con scripts y notebooks reproducibles y herramientas modernas.

La organización del repositorio está inspirada en la estructura propuesta por [cookiecutter-data-science](https://github.com/drivendataorg/cookiecutter-data-science).

```bash
.
├── data/                           # datos organizados por etapas de transformación
│   ├── raw/                        # reservado para datos originales sin procesar
│   ├── interim/                    # intents_train.csv e intents_test.csv transformados
│   └── processed/                  # splits canónicos train/dev/test listos para experimentos
├── models/                         # modelos organizados por origen
│   ├── external/                   # modelos externos preentrenados
│   └── custom/                     # modelos ajustados propios
├── notebooks/                      # notebooks de análisis
│   └── analysis/                   # análisis exploratorio y evaluación de resultados
│       ├── eda.ipynb               # análisis exploratorio de los datos
│       ├── compare.ipynb           # comparativa de resultados en dev
│       └── best.ipynb              # evaluación final del mejor modelo en test
├── prompts/                        # plantillas de prompts
├── reports/                        # documentación elaborada
├── results/                        # resultados de experimentos
├── scripts/                        # procesamiento y experimentación
│   ├── processing/                 # transformaciones de datos
│   └── experiments/                # pipelines de experimentación
├── src/intent_classification/      # código importable
│   ├── config.py                   # configuración general
│   ├── dataset.py                  # carga de datos
│   ├── evaluation.py               # evaluación de modelos
│   ├── llm/                        # cliente Ollama y entrenamiento SFT
│   └── prompting/                  # constructores de prompts
├── .vscode/                        # configuración del editor
├── pyproject.toml                  # dependencias y tooling
└── uv.lock                         # versiones fijadas
```

## Instalación

### Configuración VSCode

El proyecto está optimizado para VSCode. Al abrirlo, sugiere las siguientes extensiones:

- **Python**: soporte general de Python.
- **Pylance**: autocompletado y análisis estático.
- **Jupyter**: ejecución de notebooks.
- **Ruff**: linting y formateo rápido.

Ruff aplica las reglas de linting y formato definidas en el proyecto sincronizándose con VSCode para mantener un estilo consistente.

### Instalación de uv

**uv** es el gestor de dependencias y entornos virtuales usado en este proyecto.

La instalación de uv depende del sistema operativo:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternativamente con pip
pip install uv
```

Con uv instalado, `uv sync --extra dev` crea el entorno virtual y sincroniza las dependencias del proyecto.

Algunos de los comandos básicos de uv son los siguientes:

```bash
uv sync --extra dev # sincronizar dependencias y crear entorno virtual
uv add <paquete>    # añadir una dependencia
uv remove <paquete> # eliminar dependencias
```

Más información sobre cómo usar uv y formas alternativas de instalarlo puede encontrarse en los siguientes enlaces:

- [Documentación oficial de uv](https://docs.astral.sh/uv/)
- [Instalación de uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Guía de comandos de uv](https://docs.astral.sh/uv/reference/cli/)
- [Gestión de proyectos con uv](https://docs.astral.sh/uv/guides/projects/)

### Descarga de datos y modelos

Para la ejecución de los experimentos, es necesario descargar y procesar el dataset y disponer de un servidor [Ollama](https://ollama.com/download) local con el modelo descargado.

**1. Transformación del dataset Banking77:**

El dataset se descarga automáticamente desde [Hugging Face](https://huggingface.co/datasets/mteb/banking77) y se transforma al ejecutar los scripts en la carpeta `scripts/processing/`.

**2. Modelo en Ollama:**

Los experimentos de prompting usan `LLMClient`, que consulta el Ollama local mediante `ChatOllama` de LangChain con salida estructurada (Pydantic), sin necesidad de configuración adicional. Una vez instalado Ollama, basta con descargar el modelo:

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
```

El servidor suele arrancar solo como servicio en segundo plano. En caso contrario, se puede levantar con `ollama serve`.

El modelo se elige en el propio script mediante el argumento `model_name`, y es el que queda registrado en los resultados. Por ello, ejecutar con otro modelo genera su propia carpeta en lugar de sobrescribir la anterior:

```python
client = LLMClient(prompt, model_name='qwen2.5:3b-instruct-q4_K_M')
```

> El experimento de SFT entrena y ejecuta el modelo con transformers en el propio ordenador y requiere GPU para un uso eficiente.

## Ejecución

El proyecto incluye notebooks de análisis y exploración, así como scripts para preparar los datos y ejecutar los diferentes experimentos de clasificación de intenciones.

### Análisis y exploración

Los notebooks de análisis se encuentran dentro de `notebooks/analysis/` y se pueden abrir directamente en VSCode o a través de Jupyter.

En particular, se dispone de los siguientes notebooks:

- `analysis/eda.ipynb`: realiza un análisis exploratorio básico del dataset.
- `analysis/compare.ipynb`: compara los resultados de los diferentes experimentos en dev.
- `analysis/best.ipynb`: evalúa el mejor enfoque seleccionado en dev sobre el conjunto de test.

### Preparación de datos

Los scripts de `scripts/processing/` preparan los datos para los posteriores experimentos. Según el nivel de procesamiento, guardan los resultados de estas transformaciones en `data/interim/` y `data/processed/`.

En particular, se disponen de los siguientes scripts de preparación:

- `extract_intents.py`: descarga Banking77 desde Hugging Face, agrupa las 77 intenciones originales en 6 categorías (Account, Cards, Disputes, Payments, Topups, Transfers) y guarda los splits resultantes en `data/interim/intents_train.csv` y `data/interim/intents_test.csv`.
- `prepare_train_dev_test.py`: crea los splits canónicos train/dev/test estratificados y los guarda en `data/processed/`.

> Es importante ejecutar estos scripts en orden antes de lanzar los experimentos descritos posteriormente.

### Experimentos

Los scripts de `scripts/experiments/` comparan distintas estrategias basadas en LLMs para la detección de intenciones en consultas bancarias.

En particular, se disponen de los siguientes scripts de experimentación:

- `llm_zeroshot.py`: clasifica sin ejemplos en el prompt.
- `llm_oneshot.py`: incluye 1 ejemplo por clase en el prompt como demostración estática.
- `llm_fewshot.py`: incluye 5 ejemplos por clase en el prompt como demostración estática.
- `llm_rag.py`: selecciona dinámicamente ejemplos de entrenamiento similares a cada consulta mediante sentence-transformers.
- `llm_sft_local.py`: aplica SFT con QLoRA sobre el modelo base (`Qwen/Qwen2.5-0.5B-Instruct`) y realiza la inferencia con el modelo ajustado.

Los resultados en **train** y **dev** se guardan dentro de la carpeta `results/<nombre-del-experimento>-<modelo>/`, donde el modelo es el argumento `model_name` del cliente que lo ha ejecutado. Cada CSV incluye además una columna `model` con el identificador completo del modelo.

```
results/
├── llm-rag-qwen2.5:3b-instruct-q4_k_m/             # qwen2.5:3b-instruct-q4_K_M servido con Ollama
└── llm-sft-local-qwen-qwen2.5-0.5b-instruct/       # Qwen/Qwen2.5-0.5B-Instruct ajustado con transformers
```

Posteriormente, se pueden comparar los resultados de **dev** en el notebook `notebooks/analysis/compare.ipynb`.

> Es importante realizar la experimentación inicial y la selección del enfoque en base a **dev**. Una vez seleccionado el mejor modelo con ese split, se debe validar su rendimiento en base a **test** ejecutando el notebook `notebooks/analysis/best.ipynb`. Esta separación evita sobreajuste y mantiene la evaluación imparcial.

## Licencia

Este proyecto está licenciado bajo Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0). Ver el archivo [LICENSE](LICENSE) para más detalles.
