from src.services.batch_repository import BatchRepository
from src.services.storage_service import StorageService


def test_save_and_load(tmp_path):
    storage = StorageService(base_dir=tmp_path.as_posix())
    repo = BatchRepository(storage=storage)
    repo.save("b1", {"a": 1})
    assert repo.load("b1") == {"a": 1}


def test_results_roundtrip(tmp_path):
    storage = StorageService(base_dir=tmp_path.as_posix())
    repo = BatchRepository(storage=storage)
    repo.save_results("b2", [{"x": 2}])
    assert repo.load_results("b2") == [{"x": 2}]
