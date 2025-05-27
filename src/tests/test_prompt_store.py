import io
import types
import pytest
from src.app import prompt_store


def test_load_prompt_files_file(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("a\nb\n")
    assert prompt_store.load_prompt_files(str(p)) == ["a", "b"]


def test_load_prompt_files_directory(tmp_path):
    d = tmp_path
    (d / "a.txt").write_text("1\n2\n")
    (d / "b.txt").write_text("3\n")
    result = prompt_store.load_prompt_files(d.as_posix())
    assert set(result) == {"1", "2", "3"}


def test_load_prompt_files_missing(tmp_path):
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        prompt_store.load_prompt_files(str(missing))


def test_load_prompt_files_s3(monkeypatch):
    data = "x\ny"

    class FakeBody:
        def read(self):
            return data.encode()

    class FakeS3Client:
        def get_object(self, Bucket, Key):
            assert Bucket == "bucket"
            assert Key == "file.txt"
            return {"Body": FakeBody()}

    def fake_client(name):
        assert name == "s3"
        return FakeS3Client()

    fake_boto3 = types.SimpleNamespace(client=fake_client)
    monkeypatch.setattr(prompt_store, "boto3", fake_boto3)

    prompts = prompt_store.load_prompt_files("s3://bucket/file.txt")
    assert prompts == ["x", "y"]


def test_load_default_prompts(tmp_path):
    system = tmp_path / "sys.md"
    user = tmp_path / "usr.md"
    system.write_text("system")
    user.write_text("user")
    prompts = prompt_store.load_default_prompts(system, user)
    assert prompts.system == "system"
    assert prompts.user == "user"
