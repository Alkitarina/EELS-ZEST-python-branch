import sys
from pathlib import Path
import pytest
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def bounded_threads():
    with threadpool_limits(2):
        yield


@pytest.fixture(scope="session")
def reference():
    path = ROOT / "reference/pixel_reference.mat"
    if not path.is_file():
        pytest.skip("Private MATLAB reference fixture is not distributed; see reference/README.md")
    return loadmat(path, simplify_cells=True)


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    import os

    os.environ["ZEST_PARALLEL_RUNTIME"] = str(tmp_path_factory.mktemp("api-runtime"))
    from fastapi.testclient import TestClient
    import app

    with TestClient(app.app) as client:
        yield client
