from self_training.eval_harness import run_eval_suite
import inspect


def test_run_eval_suite_interface():
    sig = inspect.signature(run_eval_suite)
    params = list(sig.parameters.keys())
    assert "model" in params
    assert "datasets" in params
    assert "device" in params
