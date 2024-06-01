import os
from glob import glob

base_dir = os.path.join(os.path.dirname(__file__), "..")
command = f"""jupyter nbconvert \
--output-dir={os.path.join(base_dir, "python")} \
--to script {os.path.join(base_dir, "notebooks", "*.ipynb")}"""

os.system(command)

# Replace os.getcwd() -> os.path.dirname(__file__)
for script_path in glob(os.path.join(base_dir, "python", "*.py")):
    print(script_path)
    with open(script_path) as f:
        script = f.read()
    with open(script_path, "w") as f:
        f.write(script.replace("os.getcwd()", "os.path.dirname(__file__)"))
