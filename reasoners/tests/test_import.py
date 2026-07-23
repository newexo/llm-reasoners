import reasoners


def test_public_api_importable():
    for name in reasoners.__all__:
        assert hasattr(reasoners, name)


def test_core_abstractions_are_classes():
    for name in ["WorldModel", "LanguageModel", "SearchConfig", "SearchAlgorithm", "Reasoner"]:
        assert isinstance(getattr(reasoners, name), type)
