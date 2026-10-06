# 缓存、C++ 内核与保存链路验证 · 0.4.0 · 2026-10-06

本次实施保存链路、Stage 1 不变量缓存、核平移及矩阵/残差组装、Stage 2 温度扫描工作区复用。本次不更换 NNLS / BVLS / SLSQP 求解器，不研究或修改外层 Jacobian。

`zest/backend.py` 与修改前源码逐字节相同；有限差分 `jac` 的 AST 相同；`inner` 中约束、求解器选择、回退顺序、惩罚与容差的代码相同。只替换它们之前的 C/d 数据组装。没有改迭代预算、扫描网格、Float64、热启动顺序及四色空间同步。

## 实现

- `zest/acceleration.py`、`native/kernels.cpp`：每次求解私有的插值几何、矩阵缓冲区和严格 Float64 C++ 内核，通过 pybind11 接入。MSVC 使用 `/fp:strict`；其他编译器关闭 fast-math 和浮点收缩。归一化归约、矩阵乘法和线性求解仍走原路径。非均匀轴保留 NumPy/CuPy 的插值运算。
- `zest/regularization.py`、`stage1.py`：每个像素、每轮准备权重、保护曲线、静态正则和能量索引。每次只更新依赖 theta 的 BE 项、ZLP 列及尾部对称约束；fitted-ZLP 的两次线性求解均保留。目标评估仅返回所需残差，最终解仍生成完整诊断。模板和空间精修复用同一核心。
- `stage2.py`：固定 hard/soft 核、正则结构、上下界与权重只准备一次，逐温度更新 hard-BE 列和 soft-BE 正则。求解与温度选择顺序不变。
- `storage.py`、`app.py`：使用 orjson 编码有限数组，保留 NaN / ±Inf 标记及旧读写格式；float32 数组先转 Float64，保持原 `tolist()` 的数值含义。gzip 级别由 9 改为 1，整块压缩并原子替换。最终导出比较完整编码内容，仅跳过完全相同的文件，支持检测原地修改，不使用散列或对象身份猜测。ZIP 不再二次压缩 `.json.gz`。
- `hierarchy.py`：未接受且无错误的精修不再重复写像素。完整结果依旧同步保存后计为已保存；取消、恢复和最终快照语义保留。

`setup.ps1` 自动尝试构建可选 C++ 扩展，`-SkipNative` 可跳过。默认 `ZEST_CPU_KERNELS=auto`；缺少扩展时仍有 NumPy 缓存路径，可用 `numpy` 强制此路径，用 `native` 要求必须有扩展。执行记录保存实际路径，`storage` 保存真实写入数和跳过数。Web 版本更新至 0.4.0。

## 公开验证范围

这是本地加速验证记录的公开摘要。私人样本信息、原始测量记录、截图、开发机环境和编译产物不随仓库分发。本次公开副本的实际测试结果见 [PUBLICATION.md](../PUBLICATION.md)。

`tests/test_acceleration.py` 包含 C++/NumPy 内核、非均匀网格、缓存复用和结果独立性、不同 hard/soft 模态组合、soft-BE 正则、float32/NaN/±Inf/非连续数组保存、原地修改检测、损坏检查点修复、原子失败保留和原生内核参数检查。

需私人 MATLAB 参考样本的测试在公开检出中明确跳过；未构建扩展时原生内核测试也会跳过，不能将跳过计为通过。可通过 `requirements-native.txt` 与 `tools/build_native.py` 构建扩展，使用 `ZEST_CPU_KERNELS=numpy` 验证可选回退路径。

这些测试不构成所有输入的科学等价证明或整图性能基准；原 Python 与 MATLAB 端到端温度差异的限制仍见 [VALIDATION.md](../VALIDATION.md)。本仓库不据私人样本的局部耗时比承诺整图加速。
