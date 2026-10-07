"""测试包。

显式声明为包（而非依赖 pytest 的 rootdir 路径注入），
这样各测试模块可以 `from tests.conftest import ...` 复用夹具常量。
"""
