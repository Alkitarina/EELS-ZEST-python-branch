"""Job-local monotonic wall timers; nested phases never count time twice."""

from contextlib import contextmanager
from threading import RLock
import time

LABELS = {
    "prepare": "载入数据与任务准备",
    "preprocess": "预处理准备与同步",
    "align": "ZLP 对齐",
    "svd": "SVD / PCA 降噪",
    "smooth": "谱线平滑",
    "dataset_save": "保存预处理数据",
    "calibration": "稀疏网格标定",
    "pool_setup": "共享数据与工作进程启动",
    "stage1": "Stage 1 条带 / 串行拟合",
    "hierarchy": "全图聚类、模板构建与汇总",
    "template_refit": "全图模板重拟合",
    "spatial_refit": "全图空间重拟合",
    "stage2": "Stage 2 温度扫描",
    "checkpoint": "拟合中保存像素结果",
    "pool_shutdown": "工作进程退出与资源释放",
    "maps": "生成结果图",
    "export_maps": "导出 MAT / NPZ 图",
    "export_metadata": "整理结果与导出参数",
    "export_pixels": "最终保存像素结果",
    "archive": "压缩结果 ZIP",
}


class PhaseTimer:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.started = self.since = clock()
        self.ended = None
        self.stack = []
        self.seconds = {}
        self.calls = {}
        self.lock = RLock()

    def _accrue(self, now):
        if self.stack:
            key = self.stack[-1]
            self.seconds[key] += max(0.0, now - self.since)
        self.since = now

    @contextmanager
    def phase(self, key):
        with self.lock:
            if self.ended is not None:
                raise RuntimeError("计时器已结束")
            self._accrue(self.clock())
            self.stack.append(key)
            self.seconds.setdefault(key, 0.0)
            self.calls[key] = self.calls.get(key, 0) + 1
        try:
            yield
        finally:
            with self.lock:
                self._accrue(self.clock())
                self.stack.pop()

    def finish(self):
        with self.lock:
            if self.stack:
                raise RuntimeError("阶段尚未退出，不能结束计时")
            if self.ended is None:
                self.ended = self.clock()
        return self.snapshot()

    def snapshot(self):
        with self.lock:
            now = self.ended if self.ended is not None else self.clock()
            values = dict(self.seconds)
            active = self.stack[-1] if self.stack else None
            if active:
                values[active] += max(0.0, now - self.since)
            wall = max(0.0, now - self.started)
            return dict(
                version=1, clock="monotonic_wall", exclusive=True,
                complete=self.ended is not None, active_phase=active,
                wall_seconds=wall, accounted_seconds=sum(values.values()),
                other_seconds=max(0.0, wall - sum(values.values())),
                phases=[dict(key=k, label=LABELS.get(k, k), seconds=v,
                             calls=self.calls[k]) for k, v in values.items()],
                note="各项为互不重叠的墙钟时间；计算阶段含调度和等待，像素保存单列。"
                     "结果主体 ZIP 完成后停止计时，最终计时元数据写入不计入。",
            )
