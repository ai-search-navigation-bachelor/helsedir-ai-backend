"""
Quick test script to verify the setup is correct.
Run this before starting the server to check configuration.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_setup():
    """Check if the project is set up correctly."""
    print("Checking project setup...\n")

    errors = []
    warnings = []

    # Check if .env exists
    if not Path(".env").exists():
        warnings.append("WARNING: .env file not found. Using default configuration.")
        print("   Create one with: cp .env.example .env")
    else:
        print("[OK] .env file found")

    # Check if content.json exists
    if not Path("data/content.json").exists():
        errors.append("[ERROR] data/content.json not found")
    else:
        print("[OK] data/content.json found")

    # Check if required packages are installed
    try:
        import fastapi
        print("[OK] fastapi installed")
    except ImportError:
        errors.append("[ERROR] fastapi not installed. Run: pip install -r requirements.txt")

    try:
        import uvicorn
        print("[OK] uvicorn installed")
    except ImportError:
        errors.append("[ERROR] uvicorn not installed. Run: pip install -r requirements.txt")

    try:
        import pydantic
        print("[OK] pydantic installed")
    except ImportError:
        errors.append("[ERROR] pydantic not installed. Run: pip install -r requirements.txt")

    # Try to load config
    try:
        from app.config import settings
        print("[OK] Configuration loaded successfully")
        print(f"  - Host: {settings.host}")
        print(f"  - Port: {settings.port}")
        print(f"  - Environment: {settings.environment}")
    except Exception as e:
        errors.append(f"[ERROR] Failed to load configuration: {e}")

    # Try to import main app
    try:
        from app.main import app
        print("[OK] FastAPI app loaded successfully")
    except Exception as e:
        errors.append(f"[ERROR] Failed to load FastAPI app: {e}")

    print("\n" + "=" * 50)

    if errors:
        print("\nSetup has errors:")
        for error in errors:
            print(f"  {error}")
        return False

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")

    print("\nSUCCESS: Setup is complete!")
    print("\nTo start the server, run:")
    print("  python scripts/run.py")
    print("\n" + "=" * 50)

    return True


if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)
