# third_party

本目录是随仓库一起提交的第三方库源码，供 `benchmark/` 使用，内容未做修改。

| 目录 | 库 | 版本 | 来源 | 许可证 |
| --- | --- | --- | --- | --- |
| `eigen/` | Eigen | 3.4.0（tag `3.4.0`） | https://gitlab.com/libeigen/eigen | MPL2（部分文件为 BSD / LGPL / Apache / MINPACK，见 `eigen/COPYING.*`） |
| `qdldl/` | QDLDL | 0.1.9（提交 `af62a6f`） | https://github.com/osqp/qdldl | Apache-2.0（见 `qdldl/LICENSE`） |

说明：

- `eigen/` 只保留了头文件目录 `Eigen/`、许可证文件和 `README.md`；上游的测试、文档、`unsupported/`、BLAS/LAPACK 等没有包含。
- `qdldl/` 是完整的上游源码。它的 `qdldl_types.h` 和 `qdldl_version.h` 由 CMake 从 `configure/*.h.in` 生成；`benchmark/ldl_eigen_vs_qdldl/generated/` 下有一份手写的等价版本（double 精度、32 位整数索引），不依赖 CMake。
