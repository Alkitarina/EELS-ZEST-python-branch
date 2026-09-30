# 验证范围与限制

这是原本地移植验证记录的公开摘要。原记录包含私人样本信息和测量结果，未随仓库发布。

当前实现已包含选定的谱像分解与测温工作流，但尚未通过从头拟合时与 MATLAB 的温度等价性验收。固定参数下的矩阵对照、相同 Stage 1 输入下的 Stage 2 对照，不足以证明完整求解的温度一致。SciPy 与 MATLAB 的约束求解、有限差分、终止判断和局部解可能不同；不能用高 R² 代替温度精度验证。

`tests/test_numerics.py` 保留矩阵、权重、正则项、温度、分组、序列化、空间推断和取消等断言。依赖私人 `reference/pixel_reference.mat` 的测试在公开检出中明确跳过；GPU 不可用时相关测试也会跳过。公开测试的实际运行结果见 [PUBLICATION.md](PUBLICATION.md)。

运行开发测试：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -X utf8 -m pytest -q -rs -p no:cacheprovider --basetemp=.tmp_tests
node --test tests/test_network.cjs
```

可选 GPU 依赖见 `requirements-gpu.txt`。有权使用的本地 MATLAB 对照数据可按 [reference/README.md](reference/README.md) 准备。没有这些数据时不能复现原私人样本对照，也不应宣称数值等价测试通过。

支持 CUDA 不等于已证明 GPU 加速；有约束求解和外层优化仍在 CPU。未在公开数据上建立整图速度基准或整图科学精度验收。扫描谷宽沿用原算法定义，不是统计置信区间；空间推断温度不是该像素独立测得的温度。

范围限于 README 列出的工作流，不包括原 MATLAB GUI 的独立 STA、RL 反卷积和通用峰拟合等工具。
