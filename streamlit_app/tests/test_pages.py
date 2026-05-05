from pathlib import Path
from streamlit.testing.v1 import AppTest

_APP_DIR = Path(__file__).parent.parent


def test_home_loads_without_exception():
    at = AppTest.from_file(str(_APP_DIR / "app.py"))
    at.run()
    assert not at.exception
