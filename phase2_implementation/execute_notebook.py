"""I execute the research notebook with this environment's exact Python kernel."""

import json
import os
from pathlib import Path
import sys
import time


def main():
    project = Path(__file__).resolve().parents[1]
    for variable, relative in {
        "MPLCONFIGDIR": ".mplconfig", "IPYTHONDIR": ".venv/ipython",
        "JUPYTER_CONFIG_DIR": ".venv/jupyter/config",
        "JUPYTER_DATA_DIR": ".venv/jupyter/data",
        "JUPYTER_RUNTIME_DIR": ".venv/jupyter/runtime",
    }.items():
        destination = project / relative
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(destination)
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    path = project / "phase2_implementation/notebooks/pipeline_execution.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    start = time.monotonic()

    def progress(cell, cell_index, **kwargs):
        print(f"I executed notebook cell {cell_index + 1}.", flush=True)

    print("I am starting the project Python kernel.", flush=True)
    NotebookClient(notebook, km=manager, timeout=1200,
                   resources={"metadata": {"path": str(project)}},
                   on_cell_executed=progress).execute()
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code):
        raise AssertionError("An executable notebook cell was not executed.")
    if any(output.output_type == "error" for cell in code for output in cell.outputs):
        raise AssertionError("Notebook contains an execution error.")
    result = {"status": "success", "code_cells_executed": len(code),
              "elapsed_seconds": round(time.monotonic() - start, 3),
              "python_executable": sys.executable}
    notebook.metadata["phase2_execution"] = result
    nbformat.write(notebook, path)
    (project / "data/processed/phase2_notebook_execution.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
