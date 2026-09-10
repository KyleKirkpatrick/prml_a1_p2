import json

with open("PRML-Week3-Lab-MNIST_ClassificationUsingLogisticRegression.ipynb", encoding="utf-8") as f:
    notebook = json.load(f)

code = "\n\n".join(
    "".join(cell["source"])
    for cell in notebook["cells"]
    if cell["cell_type"] == "code"
)

with open("all_code.py", "w", encoding="utf-8") as f:
    f.write(code)
