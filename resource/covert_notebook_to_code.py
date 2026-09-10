import json
from pathlib import Path


script_directory = Path(__file__).resolve().parent

for notebook_path in script_directory.glob("*.ipynb"):
    with notebook_path.open(encoding="utf-8") as f:
        notebook = json.load(f)

    code = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    notebook_path.with_suffix(".py").write_text(code, encoding="utf-8")
