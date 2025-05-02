# tests/test_perf.py
import time
import torch
import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_speed(test_model):
    # Benchmark inference speed
    test_model = test_model.cuda()
    dummy_input = torch.randn(16, 16, 256, 256).cuda()  # Batch of 16
    
    # Warmup
    for _ in range(3):
        _ = test_model(dummy_input)
    
    # Timed test
    start = time.time()
    for _ in range(10):
        _ = test_model(dummy_input)
    avg_time = (time.time() - start) / 10
    assert avg_time < 0.1  # <100ms per batch

def test_memory_usage(test_model):
    # Check for OOM errors
    try:
        large_batch = torch.randn(32, 16, 256, 256)  # Stress test
        _ = test_model(large_batch)
    except RuntimeError as e:
        pytest.fail(f"OOM error: {e}")
