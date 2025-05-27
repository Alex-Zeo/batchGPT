from src.app.batch_manager import BatchManager


def test_add_and_get_batch(tmp_path):
    manager = BatchManager(storage_dir=tmp_path.as_posix())
    manager.add_batch('batch1', {'status': 'pending', 'created_at': '2024-01-01T00:00:00', 'model': 'gpt'})
    batch = manager.get_batch('batch1')
    assert batch is not None
    assert batch['id'] == 'batch1'
    assert batch['status'] == 'pending'
    assert batch['model'] == 'gpt'
