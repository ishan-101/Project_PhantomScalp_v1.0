# synthetic_data_generator/engine/utils/__init__.py
"""
Minimal utils package init to avoid circular imports.

Do NOT import submodules here. Consumers should import modules explicitly:
    from synthetic_data_generator.engine.utils import loader
    from synthetic_data_generator.engine.utils import io_writer

Keeping this file minimal prevents circular import chains.
"""

__all__ = []
