# Re-export MistralHumorScorer from the parent-level mistral_evaluator.py
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "_mistral_evaluator_module",
    str(pathlib.Path(__file__).parent.parent / "mistral_evaluator.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MistralHumorScorer = _mod.MistralHumorScorer
