"""Spatial scheduling only: never cluster spectra or cut hierarchy neighbours."""

from collections import defaultdict

PARTITION_MODES = ("balanced", "vertical", "horizontal", "grid")
MAX_BLOCKS = 10000


def validate_partition(mode, grid_rows=1, grid_columns=1):
    if mode not in PARTITION_MODES:
        raise ValueError("分块方式必须为 balanced / vertical / horizontal / grid")
    for value in (grid_rows, grid_columns):
        if type(value) is not int or not 1 <= value <= MAX_BLOCKS:
            raise ValueError("网格行数和列数需要为 1–10000 的整数")
    if mode == "grid" and grid_rows * grid_columns > MAX_BLOCKS:
        raise ValueError("网格总块数不能超过 10000")


def _split(values, count):
    """Nonempty pieces; distribute the remainder instead of dropping pixels."""
    if not values:
        return []
    count = min(count, len(values))
    size, remainder = divmod(len(values), count)
    result, start = [], 0
    for i in range(count):
        end = start + size + (i < remainder)
        result.append(values[start:end])
        start = end
    return result


def _snake(points, lookup, horizontal=False):
    lines = defaultdict(list)
    for y, x in points:
        lines[y if horizontal else x].append(x if horizontal else y)
    indices = []
    for i, line in enumerate(sorted(lines)):
        for v in sorted(lines[line], reverse=bool(i % 2)):
            indices.append(lookup[(line, v) if horizontal else (v, line)])
    return indices


def partition_blocks(coords, count, mode="balanced", grid_rows=1, grid_columns=1):
    """Return canonical indices in adjacent snake order inside each block.

    Balanced mode splits one continuous spatial snake by *pixel count*. Blocks
    may have partial columns; their sizes differ by at most one. Directional
    modes retain complete columns/rows. Grid mode makes rectangular tiles.
    Sparse selections are supported; the worker restarts across any gap.
    """
    validate_partition(mode, grid_rows, grid_columns)
    if type(count) is not int or count < 1:
        raise ValueError("块数必须是正整数")
    points = [tuple(map(int, p)) for p in coords]
    lookup = {p: i for i, p in enumerate(points)}
    if len(lookup) != len(points):
        raise ValueError("像素坐标不能重复")
    if not points:
        return []
    rows = sorted({y for y, _ in points})
    columns = sorted({x for _, x in points})
    if mode == "balanced":
        paths = _split(_snake(points, lookup), count)
    else:
        ys = _split(rows, count if mode == "horizontal" else grid_rows if mode == "grid" else 1)
        xs = _split(columns, count if mode == "vertical" else grid_columns if mode == "grid" else 1)
        y_group = {y: i for i, group in enumerate(ys) for y in group}
        x_group = {x: i for i, group in enumerate(xs) for x in group}
        groups = defaultdict(list)
        for y, x in points:
            groups[(y_group[y], x_group[x])].append((y, x))
        paths = [_snake(groups[key], lookup, horizontal=mode == "horizontal")
                 for key in sorted(groups)]
    blocks = []
    for i, indices in enumerate(paths):
        yy = sorted({points[j][0] for j in indices})
        xx = sorted({points[j][1] for j in indices})
        blocks.append(dict(id=i, indices=indices, rows=yy, columns=xx, mode=mode,
                           bounds=[yy[0], xx[0], yy[-1], xx[-1]]))
    return blocks


def partition_boundaries(coords, blocks):
    """Merged exact edges in the UI's one-based coordinates, including ROI rim.

    Bounding rectangles would misrepresent balanced blocks with partial columns.
    Merge collinear unit edges to keep job polling and Plotly payloads compact.
    """
    owners = {tuple(coords[i]): block["id"] for block in blocks for i in block["indices"]}
    edges = defaultdict(set)
    for (y, x), owner in owners.items():
        for yy, xx, axis, line, start in (
            (y, x - 1, "v", x, y), (y, x + 1, "v", x + 1, y),
            (y - 1, x, "h", y, x), (y + 1, x, "h", y + 1, x),
        ):
            if owners.get((yy, xx)) != owner:
                edges[(axis, line)].add(start)
    segments = []
    for (axis, line), positions in sorted(edges.items()):
        ordered = sorted(positions)
        start = end = ordered[0]
        for position in ordered[1:] + [None]:
            if position == end + 1:
                end = position
                continue
            segments.append([line + .5, start + .5, line + .5, end + 1.5]
                            if axis == "v" else
                            [start + .5, line + .5, end + 1.5, line + .5])
            start = end = position
    return segments
