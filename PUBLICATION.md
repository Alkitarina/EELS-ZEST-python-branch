# 公开发布检查

## 2026-10-06 · 0.4.0 同步

同步可选 C++/pybind11 Float64 内核、Stage 1/2 缓存与工作区、orjson/gzip 保存链路、构建脚本、依赖、回归测试及说明。原项目来源、作者署名、GPL-3.0-only 和公开副本的数据隔离规则保留。

- 在公开副本中从 `native/kernels.cpp` 成功编译扩展，使用 MSVC 严格浮点选项。编译产物、构建目录及私人测量 JSON 均被 Git 排除。
- 未构建扩展时，新增加速测试 **9 passed, 9 skipped, 0 failed**；其中 8 项需要可选扩展，1 项需要私人参考样本。
- 编译扩展后，完整公开 Python 测试 **28 passed, 15 skipped, 0 failed**。15 项跳过均因私人 MATLAB 参考样本未分发，不能计为数值等价性验证通过。
- JavaScript 网络恢复测试 **7 passed, 0 failed**；JavaScript 与 PowerShell 语法检查通过。
- Python 测试的三类警告分别是刻意构造 NaN/Inf 的运算提示、Starlette/httpx 弃用提示和 CuPy CUDA_PATH 探测提示。
- 待提交文本再次接受常见凭据与私人路径扫描。私人数据、运行结果和本机环境不随更新上传；原始性能测量记录仅保留在本地。

测试使用已有 Python 环境；不代表在全新机器上验证了依赖安装，也不构成整图性能或所有输入的科学等价证明。实现范围与限制见 [ACCELERATION_VALIDATION.md](docs/ACCELERATION_VALIDATION.md)。

## 2026-09-30 · 首次公开发布

本仓库由本地 `eels-zest-web-parallel` 的独立公开副本生成。未导入本地历史、私人数据或运行环境。

## 来源与许可证

来源于对 [Iridium-191/EELS-ZEST releases](https://github.com/Iridium-191/EELS-ZEST/releases) 的改写和改编，以 v2.1.0 为基础。README 和 NOTICE 保留原作者 Ruilin Mao 署名、来源链接与 GPL-3.0-only 说明；LICENSE 保留原文本。Plotly 的分发版权声明保留。

## 发布范围

- 包含 Python/Web 程序、算法默认配置、启动脚本、依赖清单、测试和公开文档。
- 排除私人 MAT/NPZ 参考光谱、原始科研数据、运行结果、日志、截图、虚拟环境、缓存及临时目录。
- 移除本机数据路径、样本名称和私人结果表；参考输入改为本地 `ZEST_REFERENCE_PATH` 可配置项。
- MATLAB 导出工具和源码检查工具接受显式本地路径，不依赖原开发机器的目录。
- `.gitignore` 排除常见凭据文件、环境文件、科学数据和生成结果。
- 待发布文件进行常见密钥/令牌、私钥、带凭据 URL、密码赋值及私人路径的静态检查；不在报告中输出匹配到的秘密值。此检查不等于对任意未知秘密形式的绝对保证。

## 本次验证

在公开副本中使用本地已有的 Python 3.12 环境执行，未复制私人参考数据：

- Python：**11 passed, 14 skipped, 0 failed**。14 项跳过均因未分发私人 MATLAB 参考样本，不能计为数值等价性验证通过。
- JavaScript 网络恢复测试：**7 passed, 0 failed**。
- `node --check static/app.js` 通过；Python 源码 AST 解析通过。
- API 烟雾验证使用临时生成的合成光谱，检查健康接口、首页、无参考样本配置、NPZ 上传、CPU 对齐/PCA 及预览。
- 测试有两项环境警告：Starlette 的 httpx 弃用提示和 CuPy 的 CUDA_PATH 探测提示；均未导致失败。

这次检查确认公开副本的上述行为，不是全新机器依赖安装验证，也不是全流程科学精度或性能基准。MATLAB 导出工具未在本次发布准备中运行。科学限制见 [VALIDATION.md](VALIDATION.md)。
