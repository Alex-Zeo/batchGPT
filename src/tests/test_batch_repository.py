from services import BatchRepository, StorageService


def test_repository_roundtrip(tmp_path):
    storage = StorageService(tmp_path.as_posix())
    repo = BatchRepository(storage)

    repo.save_metadata("b1", {"x": 1})
    assert repo.load_metadata("b1") == {"x": 1}

    repo.save_results("b1", [{"y": 2}])
    assert repo.load_results("b1") == [{"y": 2}]
