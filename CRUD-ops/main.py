import subprocess

def run_script(name):
    print(f"\nRunning {name}...")
    subprocess.run(["python3", f"{name}.py"])

if __name__ == "__main__":
    for operation in ["insertion", "deletion", "query"]:
        run_script(operation)
