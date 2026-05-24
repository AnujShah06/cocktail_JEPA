"""Smoke tests for the project station. Expand per stage."""

def test_package_imports():
    import cocktail_jepa
    assert cocktail_jepa.__version__

def test_config_loads():
    from cocktail_jepa.config import CONFIG
    assert CONFIG.device in ("cuda", "mps", "cpu")
    assert CONFIG.paths.root.exists()


def test_logging_layer():
    """The logger works in console mode and never raises."""
    from cocktail_jepa.logging import get_logger
    lg = get_logger(run_name="pytest", use_wandb=False)
    lg.log({"loss": 0.5}, step=1)
    lg.summary(final=1.0)
    lg.finish()
