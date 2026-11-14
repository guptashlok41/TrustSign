import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parent

for file in ROOT.glob("*.py"):
    if file.name=="inspect_modules.py":
        continue

    print(f"\n{'='*50}")
    print(file.name)
    print("="*50)

    try:
        tree=ast.parse(file.read_text(encoding="utf-8"))

        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                print("Function:",node.name)
            elif isinstance(node,ast.ClassDef):
                print("Class:",node.name)
            elif isinstance(node,ast.If):
                condition=ast.unparse(node.test)
                if "__name__" in condition:
                    print("Main guard:",condition)
    except Exception as e:
        print("Error:",e)