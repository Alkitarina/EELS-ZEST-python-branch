from pathlib import Path
import os, re, sys
path = os.environ.get('ZEST_UPSTREAM_SOURCE')
if not path:
    sys.exit('Set ZEST_UPSTREAM_SOURCE to EELS_PearsonSparse_FullFit.m; see reference/README.md')
source = Path(path).read_text(encoding='utf-8-sig')
starts=list(re.finditer(r'^        function\b.*',source,re.M))
for i,m in enumerate(starts):
    head=m.group(0)
    if any(re.search(r'\b'+re.escape(arg)+r'\s*\(',head) for arg in sys.argv[1:]):
        print(source[m.start():starts[i+1].start() if i+1<len(starts) else len(source)])
