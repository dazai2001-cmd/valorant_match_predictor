import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from web.app import app


if __name__ == "__main__":
    log_path = PROJECT_ROOT / "logs" / "flask-app.log"
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        sys.stdout = log
        sys.stderr = log
        try:
            print("Starting Flask app on http://127.0.0.1:5000", flush=True)
            app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
        except Exception:
            traceback.print_exc()
            raise
