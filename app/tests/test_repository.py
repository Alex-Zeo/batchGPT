from app.batch_manager import BatchJob, BatchManager
from repositories.batch_repository import BatchRepository
from services import StorageService
from services.observer import CallbackObserver
from factories.provider_factory import LLMProviderFactory


def test_batch_repository_save_load(tmp_path):
    storage = StorageService(tmp_path.as_posix())
    repo = BatchRepository(storage)
    job = BatchJob(batch_id="b1", status="pending")
    repo.save(job)
    loaded = repo.load("b1")
    assert loaded is not None
    assert loaded.id == "b1"
    assert loaded.status == "pending"


def test_llm_provider_factory():
    client = LLMProviderFactory.create("openai", model="gpt-3.5-turbo")
    from app.openai_client import AsyncOpenAIClient

    assert isinstance(client, AsyncOpenAIClient)


def test_batch_manager_observer(tmp_path):
    storage = StorageService(tmp_path.as_posix())
    repo = BatchRepository(storage)
    manager = BatchManager(storage_dir=tmp_path.as_posix(), repository=repo)
    events = []
    manager.register_observer(
        CallbackObserver(lambda bid, status: events.append((bid, status)))
    )
    manager.add_batch("b1", {"status": "pending"})
    manager.update_batch("b1", status="completed")
    assert events == [("b1", "pending"), ("b1", "completed")]
