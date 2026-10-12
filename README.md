# EELS-ZEST Web Parallel

当前版本：**1.1.0**。新增启动时 / 手动设备检测、线程数减 2 的默认并行配置，以及像素均分、横切和二维网格调度。版本变更见 [CHANGELOG](CHANGELOG.md)。界面采用浅彩渐变背景、小圆角与半透明磨砂玻璃分区；阶段计时默认折叠，点击标题可展开明细，进度刷新保留展开状态。

## 项目来源与改编声明

本项目来源于对 [Iridium-191/EELS-ZEST 发布版本](https://github.com/Iridium-191/EELS-ZEST/releases) 的改写和改编，以 **EELS-ZEST v2.1.0** 的 MATLAB 工作流为基础，使用 Python / NumPy / SciPy 实现选定的谱像分解与测温流程，并增加 Web 界面、CPU 多进程调度、阶段计时及任务恢复状态显示。

- 原项目：[Iridium-191/EELS-ZEST](https://github.com/Iridium-191/EELS-ZEST)
- 对应版本：[v2.1.0](https://github.com/Iridium-191/EELS-ZEST/releases/tag/v2.1.0)
- 原作者：Ruilin Mao（International Center for Quantum Materials, Peking University）
- 本改编仓库：[Alkitarina/EELS-ZEST-python-branch](https://github.com/Alkitarina/EELS-ZEST-python-branch)
- 许可证：**GPL-3.0-only**，保留原项目版权与署名，详见 [LICENSE](LICENSE) 和 [NOTICE.md](NOTICE.md)。这是独立改编版本，不代表原作者发布或认可。

This project is a Python/Web rewrite and adaptation of selected workflows from [EELS-ZEST releases](https://github.com/Iridium-191/EELS-ZEST/releases), based on v2.1.0 by Ruilin Mao. It retains the upstream GPL-3.0-only license and attribution. It is an independent adaptation, not an official upstream release.

公开仓库仅包含程序、配置默认值、测试与文档，不包含私人科研数据、参考光谱、运行结果、日志或本机环境。首次使用请按下文安装并导入自己有权使用的数据。

`eels-zest-web` 的独立并行版本，支持 **可配置 CPU 工作进程 + 动态领取分块 + 块内相邻蛇形热启动**。默认进程数为 `min(61, max(1, 本机逻辑线程数 − 2))`，自动块数为 `max(1, 3 × 进程数 − 4)`，再按像素数 / 切分方向限制。默认沿连续蛇形路径按像素均分，也支持竖向条带、横向条带和手动二维网格。界面可切换到原列优先串行模式作对照。目标函数、约束、正则项、温度扫描和 Float64 精度保持原实现。

新版本拥有独立 `.venv`、`runtime` 和端口 **8878**，不读取或覆盖旧版的任务目录。自 2026-10-09 起，Web 版本只维护 parallel 版本；普通 `eels-zest-web` 保留为历史版本，停止维护。串行对照仍可在 parallel 界面中选择。

**当前是数值移植验证版，不应把温度结果视为已与 MATLAB 等价。** 固定参数的设计矩阵与正则矩阵已逐项对照；Stage 2 使用相同 Stage 1 输入得到相同温度。从头拟合时，SciPy 与 MATLAB 的求解轨迹存在明显温度差异。原移植验证记录见 `VALIDATION.md`；本次并行验证见 `PARALLEL_VALIDATION.md`。

## 启动

先按下方“首次安装”步骤创建独立 `.venv`，再双击 **launch.bat**，或在 PowerShell 中运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_web.ps1
```

默认网址 `http://127.0.0.1:8878`；只监听本机。从 0.3.1 起，后端独立在后台运行，启动脚本确认服务就绪后才打开浏览器；启动窗口关闭不会停止计算。停止服务请运行下述 `stop_web.ps1`。端口占用时加 `-Port 8879`，不打开系统浏览器可加 `-NoBrowser`。

每次启动的标准输出及错误日志保存在 `runtime/logs/server-端口-时间.*.log`，`runtime/logs/latest-端口.json` 记录最近启动的日志位置。Python 及计算子进程启用 `faulthandler`，可将支持捕获的底层异常栈写入错误日志；强制杀进程、断电等情况仍可能没有异常栈。日志目录随 `ZEST_PARALLEL_RUNTIME` 改变。若启动未就绪，脚本会报告日志位置；`-StartupTimeoutSeconds` 可调整默认 60 秒的启动等待时间。

网页遇到连接中断时会显示重启步骤并自动重连，不再只显示 `Failed to fetch`。重连会同步计算记录，导入和计算请求不会自动重复提交。重启后未完成任务标为「已中断」，依据磁盘检查点显示已保存像素数；这些像素可以查看，但不代表后续空间拟合和温度扫描已完成，也不会自动续跑。

停止后端可运行同目录的 `stop_web.ps1`。脚本核对端口、启动命令和本项目的 Python 环境，有计算任务时先请求取消并等待已完成结果保存（默认最多 900 秒），随后终止后端及其子进程。未运行时直接退出，不删除数据或结果。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\stop_web.ps1
# 自定义端口；只查看目标，不执行停止：
.\stop_web.ps1 -Port 8879 -WhatIf
# 后端卡死时立即终止，不等待保存：
.\stop_web.ps1 -Force
```

`-TimeoutSeconds` 可调整保存等待时间。`-Force` 仍需通过进程身份核对，但未保存的结果可能丢失。

升级时，先等待当前任务结束，运行 `stop_web.ps1`，再双击 `launch.bat` 并刷新网页。仅刷新网页不会更新 Python 后端；新界面识别不支持并行配置的旧后端时会提示重启，并暂停提交新计算，已有记录仍可查看。

首次安装（在仓库根目录执行）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
# 无 NVIDIA CUDA GPU 的机器：
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -CPUOnly
```

需要 Python 3.12 或 uv。GPU 环境采用 CuPy 14 + pip CUDA 12 工具库，需要兼容的 NVIDIA 驱动。依赖安装在项目自己的虚拟环境中。界面脚本、图表资源均本地提供，不使用 CDN。

## 使用

1. 点击「导入数据」选择本地 MAT / NPZ。支持 MATLAB `EELS` 类对象，也支持含 `data` 和 `ene` 的结构体；默认能量单位为 meV，不会自动猜测或转换 eV。
2. 按实际流程做 ZLP 对齐和 SVD/PCA。默认 rank 12，可自行修改；这是评估时使用的假设值，不是从数据识别出的最佳 rank。每次预处理产生新数据集，本次导入的原始数据及其预处理版本仍可在下拉框选择。重新导入文件（包括同名文件）后，下拉框、计算记录、任务状态和诊断结果重置为新数据组；刷新页面或自动重连仅恢复当前数据组。旧数据和结果仍保存在磁盘，不会因界面清空而删除。
3. 点击热图选择像素，先运行一次「Stage 1 + Stage 2」。也可选择矩形区域或整个图像。界面坐标从 1 开始，内部 API 从 0 开始。选择「分块并行 · 动态领取」，设置工作进程数、分块方式及总块数（0 为自动）；二维网格分别填写沿 Y 的行数和沿 X 的列数。要复现旧版分条方式，可设「竖向条带」、6 进程、6 块。单像素直接计算。设置只作用于新任务，并在此浏览器中记忆。
4. Stage 1 独立结果可从计算记录选中，然后继续 Stage 2。单像素不触发整图标定；多像素按原版条件启用稀疏网格标定和聚类模板；空间 ZLP 重拟合只在整图采样时启用。
5. 查看谱线、残差、模态、Attention、温度扫描，以及原始温度、有效温度、联合温度等图。联合图保留独立拟合温度，仅按原版用固定有效锚点推断缺失像素，推断像素单独标记。
6. 下载 ZIP：包含 `maps.mat`、`maps.npz`、参数、概要、聚类信息和每像素 `.json.gz` 结果。结果在 `runtime/jobs/<id>/`，数据在 `runtime/datasets/`。重启可查看已完成记录；中断的任务不会自动继续计算。取消保存已完成像素。并行版按原坐标索引保存，未完成位置不会被其他条带的结果覆盖。

前端维护检查使用 `node --test tests/test_network.cjs tests/test_workspace.cjs`，覆盖连接恢复、导入失败、同名文件重新导入、预处理版本分组、记录清空和异步旧请求隔离。同一数据组内快速切换计算记录、由记录切换到预处理数据时，也按最后一次选择生效。

数据集内存缓存按最近使用顺序淘汰，最多保留 3 份、数组合计不超过 512 MiB；单份超限时仅供当前请求或任务使用。积分强度图按数据集 ID 单独缓存，最多 8 份、合计 32 MiB，因此切换像素无需重复对整张谱像求和。新导入和每次预处理都有独立 ID，不会误用上一组数据的强度图。

任务成功保存最终结果及元数据后，释放内存中的完整像素结果和结果图；查看历史结果或继续 Stage 2 时从磁盘读取。计算已生成的最终结果图直接供导出复用；取消或中途失败的任务仍按已完成像素生成部分结果。缓存淘汰不删除磁盘文件，也不影响运行中任务已持有的数据。导出失败时保留内存结果，避免丢弃尚未可靠保存的内容；上述缓存额度不代表整个进程的内存上限。

## 并行方式与对照

- 本机配置仅在服务启动时检测，或点击顶部「重新检测」手动更新；没有周期硬件监测。显示 CPU 型号、核心 / 逻辑线程数、物理内存总量、GPU / CUDA 信息和检测时间。跟随推荐时，新检测会更新默认进程数；手动进程数保留，点击「恢复推荐值」可重新跟随。运行中的任务不被改写。
- 整图预处理及稀疏模态标定保持原流程，所有条带共享标定设置。
- 默认「像素均分」：先构建相邻蛇形路径，再按像素数切成连续段。各块大小最多相差 1 个像素，可以包含不完整列，不要求矩形或整除宽度。例如检测为 20 线程时推荐 18 进程；48 × 36 像素自动切为 50 块，每块 34 或 35 像素。
- 「竖向条带」按完整 X 列切分，「横向条带」按完整 Y 行切分；实际块数分别不超过所选列 / 行数。「二维网格」按指定的行数 × 列数切分，行列数分别限制到 ROI 尺寸，余数分散到前面的行 / 列组。例如 3 × 7 像素切为 2 行 × 3 列，块大小为 6、4、4、3、2、2。网格模式忽略总块数，横切块内使用横向蛇形，网格块内使用纵向蛇形。
- 进程从共享队列领取任务，完成后继续领取下一块，不绑定物理核心。`3n − 4` 是负载均衡与初始化成本之间的启发式起点，不是测得的最优值。更多分块会增加完整初始化次数，也会改变热启动链，需同时对比耗时与拟合结果。
- 每条带首像素保留全部原有优化轮次，随后继承相邻像素的 theta。失败或坐标不相邻时重新完整初始化。不开启新的相似度阈值或近似求解策略。
- 工作进程通过共享内存只读访问光谱和能量轴，每个进程限制 1 个 BLAS 线程；主服务负责进度、保存和导出。工作池最多为请求进程数与像素数的较小值，首轮最多同时计算「工作进程数、实际条带数」两者中的较小值；后续全图重拟合和 Stage 2 可使用整个池。Windows 进程数上限为 61。
- 全部 Stage 1 完成后，基于全图统一建立聚类、模板和邻域。模板重拟合及每一颜色组内并行；颜色组之间、空间扫描轮次之间等待完成。条带边界不构成新的空间屏障；原算法识别的真实界面屏障照常生效。
- 原版只在整个图像采样时开启空间扫描；矩形 ROI 保留原行为，仅使用聚类模板。要检查跨条带的全图空间拟合，请选择「整个光谱图像」并开启层级空间选项。
- Stage 2 按像素分配给同一工作池；不会重新计算 Stage 1。也支持从串行 Stage 1 记录继续并行 Stage 2。
- 串行模式保留原列优先顺序、整条热启动链和 2 个数值库线程，不创建像素进程池。
- 任务状态显示各块完成数、Y/X 范围、PID、当前像素及分阶段耗时。热图虚线按实际像素归属绘制，包括非矩形块中的横向边缘，不代表样品物理界面。旧竖条记录仍可显示。
- ZIP 包含 `execution.json` 和 `timings.json`，记录进程、遍历、条带及耗时；每像素结果的 `execution` 记录 Stage 1 热启动来源。NPZ 同时保存实际完成的 `coords` 和零基 `pixel_indices`，便于读取取消后的非连续结果。

## 阶段计时与 API

运行面板实时显示 ZLP 对齐、SVD/PCA、标定、工作进程启动、Stage 1、全图空间拟合、Stage 2、图生成、像素保存、MAT/NPZ 导出、ZIP 压缩和资源释放。串行与并行使用同一套计时口径；串行穿插执行的 Stage 1/2 会分别累加。取消或失败也保留已发生的时间。

`timings` 使用单调墙钟时间，阶段互不重叠。例如主进程写像素文件时暂停该主进程的拟合阶段计时，将这段时间计入保存；此时后台工作进程仍可能继续计算。这些值用于说明主任务执行过程，不是各 CPU 核的计算时间或独占瓶颈占比。各项与 `other_seconds` 相加等于 `wall_seconds`。计时截止到结果主体 ZIP 完成，最后很小的计时元数据写入和 ZIP 提交不计入，页面总任务耗时则包含这些工作。

新计时随 `job.json` 持久化，并写入 `timings.json` 和 ZIP 内的 `summary.json`；重启后可查看。旧任务没有完整计时，仍显示原有阶段记录，不补造缺失数据。`execution.phase_seconds` 保留早期包含回调保存的粗粒度计时口径，不能与新 `timings` 相加。

`POST /api/fit` 的调度参数放在请求顶层，不放入算法 `settings`：`parallel_workers`（1–61 的整数，省略时取最近一次检测的推荐值）、`stripe_count`（保留旧字段名，表示 0–10000 的总块数，0 为自动）、`partition_mode`（默认 `balanced`；另支持 `vertical` / `horizontal` / `grid`）、`grid_rows` 与 `grid_columns`（默认 1，正整数，总网格数不超过 10000）。例如 `{"parallel_workers":18,"stripe_count":0,"partition_mode":"balanced"}`，或 `{"parallel_workers":4,"partition_mode":"grid","grid_rows":3,"grid_columns":5}`。串行 / 单像素模式忽略分块配置，继续 Stage 2 只使用进程数，不重复 Stage 1。GPU 后端仍使用串行混合路径。

`GET /api/config` 和 `/api/health` 读取最近一次检测结果，不触发新检测。`POST /api/hardware/refresh` 重新检测并返回更新后的配置。Windows 详细硬件查询失败时保留逻辑线程数和 CPU 回退信息，界面会显示检测未完成。检测不修改系统设置。

**调度分块不等于按谱线相似度聚类。** 前者用空间相邻性保持热启动连续；真正的聚类 / 模板和空间约束仍在完整 Stage 1 后基于全图统一建立。没有新增把远距离相似谱线拼成一组的近似。每块首像素、失败后的像素和跨坐标间隙的像素仍执行完整初始化。验证范围见 [分块与设备检测验证](docs/BLOCK_PARTITION_VALIDATION.md)。

**两模式的初值传播顺序不同，最终局部解和温度不保证完全相同。** 比较时应检查温度、Anti-Stokes 系数、残差、失败率和条带边界；单凭 R² 不能证明温度一致。本次没有用整幅真实数据测量加速倍数。

API 在 `/api/fit` 请求顶层使用 `"execution_mode": "stripes"`（默认）或 `"serial"`，与科学 `settings` 分开。旧 MATLAB 的 `datacube_parallel_*` 参数不控制此调度器。并行拟合要求 `device="cpu"`；GPU 混合拟合保留在串行模式，GPU 预处理仍可用。

运行目录覆盖变量为 `ZEST_PARALLEL_RUNTIME`；忽略旧版 `ZEST_RUNTIME`，并拒绝把新运行目录指向旧版 `runtime`。关闭服务会请求工作进程停止并回收共享内存；停止按钮会保存已完成像素，不提供断点自动续跑。

## 缓存与原生内核加速

2026-10-06 起，Stage 1 按像素及优化轮次缓存谱线权重、保护曲线、插值几何和正则结构，复用设计矩阵与线性子问题的输入缓冲区。外层目标评估只构造所需残差，最终结果仍生成完整诊断。Stage 2 缓存 hard/soft 基底、正则结构、上下界及权重，按原顺序逐温度更新依赖温度的项。缓存不跨像素共享。

可选 C++ 扩展通过 pybind11 执行均匀网格核平移、加权矩阵、残差与 hard-BE 列组装，使用 Float64、严格浮点选项、不增加线程；归一化和 BLAS 运算保留原归约路径。**NNLS / BVLS / SLSQP 求解器、约束与容差、有限差分 Jacobian、温度扫描网格及热启动顺序未改变。** 模板和空间重拟合自动复用新的 Stage 1 内核。

`setup.ps1` 默认尝试安装构建依赖并编译本机扩展，需要 MSVC C++ Build Tools；可用 `-SkipNative` 跳过。扩展不可用时仍能使用缓存后的 NumPy 路径。已有环境可单独构建：

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt -r requirements-native.txt
.venv\Scripts\python.exe -X utf8 tools\build_native.py
```

`ZEST_CPU_KERNELS=auto`（默认）自动使用已构建扩展；`numpy` 强制 NumPy 缓存路径，`native` 强制 C++ 并在扩展缺失时报错。任务 `execution.acceleration` 记录实际路径。编译产物只适用于对应操作系统、架构和 Python 版本，跨机器应重新构建。GPU 保留 CuPy 路径。

像素结果仍为兼容旧版的 `.json.gz`：原生 JSON 编码保留 NaN / ±Inf 标签和浮点往返值，整块写入 gzip（压缩级别 1），继续采用临时文件原子替换。相较此前压缩级别 9，文件通常稍大；ZIP 直接收录已压缩的像素文件。未接受且无错误的空间重拟合不再重复保存；最终导出逐项比较编码内容，仅跳过完全未变化的文件，支持原地修改的检测。保存仍为同步完成后记为已保存，不改变恢复与取消的语义。任务 `storage` 记录写入和跳过数量。

数值与性能验证见 [ACCELERATION_VALIDATION.md](docs/ACCELERATION_VALIDATION.md)。

## CPU 与 GPU 的范围

| 部分 | CPU | GPU 混合后端 |
|---|---|---|
| MAT 读取、ZLP 对齐、流程控制 | NumPy / Python | CPU |
| SVD/PCA | SciPy 截断 SVD；退化矩阵使用完整 SVD | CuPy economy SVD 后保留指定 rank |
| Pearson / Pseudo-Voigt、平移核与设计矩阵 | NumPy + 可选 C++ 内核、每轮缓存 | CuPy Float64、每轮缓存 |
| 模型矩阵乘法、QP 的矩阵乘法 | CPU | CUDA |
| 非负/有界最小二乘、一般约束 QP | SciPy | SciPy / CPU |
| 外层非线性优化与温度扫描调度 | SciPy / Python | SciPy / Python |

这是**真实 CUDA 运算的混合版本**，没有把 CPU 运算伪装为 GPU。GPU 不可用会报错，不能静默退回 CPU。单像素矩阵较小且频繁传输，当前没有证明 GPU 带来速度优势；CPU 是界面默认值。CPU 多进程不会改进这条 GPU 混合路径；这里没有实现全 CUDA 的 MATLAB `lsqlin` 替代品。

## 对原版的对应关系

- `zest/stage1.py`：多分量非对称 Pearson VII / Pseudo-Voigt、fitted-ZLP 双次内层拟合、Bose–Einstein 软约束、模态平滑/L2/L1、ZLP 保护、核心和远尾锚定、原版迭代轮次与热启动。
- `zest/stage2.py`：原版 Attention、hard/soft 模态划分、有限 ZLP 修正、有界线性子问题、温度自适应网格与细化、剖面目标谷宽。
- `zest/calibration.py`、`hierarchy.py`：原版稀疏网格标定、ZLP 特征、PCA、聚类/模板、界面屏障及四色空间重拟合。
- `zest/maps.py`：有效性筛选和不传播推断值的固定锚点温度推断。
- `zest/partition.py`、`parallel.py`、`parallel_pipeline.py`：空间分块、CPU 进程池、相邻蛇形热启动及阶段调度。
- `zest/host_info.py`、`backend.py`：本机硬件清单与 CPU / CUDA 后端检测。
- `zest/cache.py`：数据集与积分强度图的有界缓存。
- `app.py`：本地服务、单任务管理、串行/并行选择、取消、持久化与导出。
- `static/`：网页工作台与本地图表资源。

MATLAB 的专有优化器改用 SciPy，因此迭代步、终止判断、约束求解数值误差和非线性局部解会不同；聚类伪随机序列也不保证跨语言相同。保留的原版自适应扫描、早停、稀疏标定不是本次新加的优化。Python 默认不启用原版可选的近似 `fast_nnls` 路径。

内层 `auto` 遵循源代码的 `lsqlin → lsqnonneg → quadprog` 顺序；L1 项在 `quadprog` 分支才参与目标。原版 `lsqnonneg` 回退会忽略尾部不等式并裁剪边界，本版保留该行为并在每像素 `inner_solver` 中记录 `original_method`、失败原因和尾部超限值。

范围是上述谱像分解和测温工作流；原 GUI 的其他工具（如独立 STA、RL 反卷积、通用峰拟合等）没有迁移。参数 JSON 用于已迁移的算法配置，MATLAB 的并行池、绘图、保存回调等参数不属于 Python 后端的可执行功能。

## 验证与开发

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=.tmp_tests
node --test tests/test_network.cjs tests/test_workspace.cjs
.venv\Scripts\python.exe -X utf8 tools\validate_native.py
```

公开测试不需要私人参考数据；依赖 `reference/pixel_reference.mat` 的测试会在该文件缺失时明确跳过。`tools/validate_native.py` 也需要该本地文件，默认检查 CPU，可加 `--devices cpu gpu` 检查 GPU。这不表示跳过的数值对照已通过。

可选参考导出与原 MATLAB 源码检查说明见 [reference/README.md](reference/README.md)。MATLAB 只用于生成对照基准，不参与 Python 程序运行。

如需启用「载入参考数据」，启动前设置 `$env:ZEST_REFERENCE_PATH = 'C:\data\example_data.mat'`，或在本地放置 `reference/example_data.mat`。这些文件均被 Git 忽略。

GPL-3.0-only，见 LICENSE 与 NOTICE.md。
