import importlib

modules = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "torch",
    "tensorflow",
    "sklearn",
    "cv2"
]

print("Checking installed libraries...\n")

for module in modules:
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "unknown")
        print(f"[OK] {module} (version: {version})")
    except ImportError:
        print(f"[MISSING] {module}")

print("\nCheck complete.")
input("Press Enter to exit...")