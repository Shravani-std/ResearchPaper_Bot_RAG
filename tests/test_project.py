import importlib


def test_core_config_and_modules_import():
    config_module = importlib.import_module("core.config")
    assert config_module.settings is not None

    retrieval_module = importlib.import_module("src.retrieval")
    llm_module = importlib.import_module("src.llm")
    ingestion_module = importlib.import_module("src.data_ingestion")

    assert hasattr(retrieval_module, "Retriever")
    assert hasattr(llm_module, "GeminiLLM")
    assert hasattr(ingestion_module, "ingest_document")
