# CUDA Python Stubs

Typed stubs for CUDA Python bindings.

This repository provides `.pyi` files for CUDA Python packages/bindings that do not ship complete type information, enabling accurate type hints and static type checking with tools such as Pylance and BasedPyright.

## Usage

Clone or fork this repository and add the typings/ directory to your Python project's type-checker configuration.

The stubs mirror the corresponding Python module hierarchy. For example:

```text
typings/
└── nvidia/
    └── nvcomp/
        └── nvcomp_impl.pyi
```

provides type information for:

```python
import nvidia.nvcomp.nvcomp_impl
```

The actual CUDA Python package must be installed separately.

## Supported Packages

- [nvCOMP](https://docs.nvidia.com/cuda/nvcomp/installation.html#installing-the-nvcomp-library-through-pypi)

## License

See [LICENSE](LICENSE).
