// Run with: node --test tests/test_network.cjs tests/test_workspace.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const config = {parallel_workers:18,max_parallel_workers:61,logical_cpus:20,auto_stripes_per_worker:3,auto_stripes_offset:4,partition_modes:['balanced','vertical','horizontal','grid'],hardware:{gpu:false},
  defaults:{crop_range:[-10,10],zlp_model_type:'pearson',stage2_T_grid:[100,300]}};

function dataset(id,parent_id) {
  return {id,parent_id,name:parent_id?'same.mat · 已预处理':'same.mat',source:'same.mat',
    shape:[2,3,4],pixels:6,energy_min:-10,energy_max:10,energy_step:5,
    history:parent_id?[{zlp_alignment:true,svd_rank:1,smoothing_points:1}]:[]};
}
function job(id,dataset_id,extra={}) {
  return {id,dataset_id,kind:'fit',status:'completed',request:{stage:'both'},
    device:'cpu',created_at:1,elapsed_seconds:5,done:1,total:1,total_pixels:1,
    completed_pixels:1,coords:[[0,0]],events:[],message:'已完成',download_ready:true,...extra};
}
const response = (payload,status=200) => ({ok:status>=200&&status<300,status,json:async()=>payload});

function harness(db,override) {
  const nodes=new Map(),saved=new Map();
  function element(tagName='div') {
    const classes=new Set();
    return {tagName,textContent:'',value:'',disabled:false,style:{},dataset:{},children:[],options:[{},{}],
      classList:{add:x=>classes.add(x),remove:x=>classes.delete(x),contains:x=>classes.has(x),
        toggle:(x,on)=>{if(on??!classes.has(x))classes.add(x);else classes.delete(x);}},
      append(...children){this.children.push(...children);},
      replaceChildren(...children){this.children=children;this.options=children;},
      setAttribute(){},on(){},removeAllListeners(){},focus(){},close(){},showModal(){},scrollIntoView(){},
      closest(){return this.parent ||= element();},
      querySelector(tag){for(const child of this.children){if(child.tagName===tag)return child;const match=child.querySelector?.(tag);if(match)return match;}return null;}};
  }
  const get=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
  get('notice').append(element('span'),element('button'));
  for(const [id,value] of Object.entries({'map-select':'intensity',scope:'pixel','fit-stage':'both',
    'execution-mode':'stripes','parallel-workers':'18','stripe-count':'0','partition-mode':'balanced','grid-rows':'1','grid-columns':'1',backend:'cpu',
    'pixel-y':'1','pixel-x':'1','roi-y0':'1','roi-x0':'1','roi-y1':'2','roi-x1':'3',
    'svd-rank':'1','smooth-points':'1'}))get(id).value=value;
  get('plot-tabs').children=['spectrum','temperature'].map(view=>Object.assign(element(),{dataset:{view}}));
  const fetch=async (url,options={})=>{
    db.calls ||= [];db.calls.push([url,options.method]);
    const custom=override?.(url,options);if(custom!==undefined)return await custom;
    if(url==='/api/config')return response(config);
    if(url==='/api/health')return response({status:'ok'});
    if(url==='/api/datasets')return response(db.datasets);
    if(url==='/api/jobs')return response(db.jobs);
    if(url==='/api/import'||url==='/api/upload'){
      if(db.importError)return response({detail:'无法读取文件'},400);
      db.datasets=[db.imported,...db.datasets.filter(d=>d.id!==db.imported.id)];return response(db.imported);
    }
    if(url==='/api/preprocess')return response(db.preprocess);
    const preview=url.match(/^\/api\/datasets\/([^/]+)\/preview/);
    if(preview)return response({energy:[-10,-5,0,5],spectrum:[1,2,3,2],intensity:[[1,2,3],[4,5,6]]});
    const data=url.match(/^\/api\/datasets\/([^/]+)$/);
    if(data)return response(db.datasets.find(d=>d.id===data[1]));
    const maps=url.match(/^\/api\/jobs\/([^/]+)\/maps$/);
    if(maps)return response({R2:[[.99,.99,.99],[.99,.99,.99]]});
    if(url.includes('/pixel?'))return response({available:false});
    const status=url.match(/^\/api\/jobs\/([^/]+)$/);
    if(status)return response(db.jobs.find(j=>j.id===status[1]));
    throw new Error(`Unexpected request: ${url}`);
  };
  const ctx=vm.createContext({fetch,AbortController,setTimeout,clearTimeout,console,FormData,structuredClone,
    document:{getElementById:get,createElement:element,createTextNode:x=>x,addEventListener(){},querySelectorAll:()=>[]},
    Plotly:{react(id,data,layout){db.plots||={};db.plots[id]={data,layout};},purge(){}},localStorage:{getItem:k=>saved.get(k)||null,setItem:(k,v)=>saved.set(k,v)}});
  vm.runInContext(source,ctx);ctx.fixtureConfig=config;
  const run=code=>vm.runInContext(code,ctx);
  run('S.config=fixtureConfig;S.initialized=true;S.online=true;');
  return {run,get,saved,json:code=>JSON.parse(JSON.stringify(run(code)))};
}

test('dataset history follows parent IDs and excludes earlier imports with the same filename',async()=>{
  const db={datasets:[dataset('old-child','old'),dataset('new-grandchild','new-child'),
    dataset('old'),dataset('new-child','new'),dataset('new')],jobs:[]};
  const h=harness(db);
  await h.run('refreshDatasets();');
  assert.equal(h.get('dataset-select').options.length,1);
  await h.run('selectDataset("new-grandchild")');
  assert.deepEqual(h.json('[...S.datasetIds].sort()'),['new','new-child','new-grandchild']);
  assert.equal(h.run('S.datasetRoot'),'new');
  assert.equal(h.get('dataset-select').options.length,4);
});

test('successful reimport resets history, records, progress, download, results and Stage 2 selection',async()=>{
  const db={datasets:[dataset('old-child','old'),dataset('old')],jobs:[job('old-fit','old-child')],imported:dataset('new')};
  const h=harness(db);
  await h.run('refreshDatasets()');await h.run('selectDataset("old-child")');
  await h.run('refreshJobs()');await h.run('selectJob("old-fit")');
  h.run('S.result={T:300};S.pixel=[1,2];S.view="temperature";S.handled.add("old-fit");$("fit-stage").value="stage2";');
  await h.run('importPath("same.mat")');
  assert.deepEqual(h.json('[...S.datasetIds]'),['new']);
  assert.equal(h.get('dataset-select').options.length,2);
  assert.equal(h.get('preprocess-history').children.length,0);
  assert.deepEqual(h.json('S.jobs'),[]);
  assert.equal(h.run('S.job'),null);
  assert.equal(h.run('S.result'),null);
  assert.deepEqual(h.json('S.maps'),{});
  assert.deepEqual(h.json('S.pixel'),[0,0]);
  assert.equal(h.run('S.view'),'spectrum');
  assert.equal(h.get('fit-stage').value,'both');
  assert.equal(h.get('job-status').textContent,'就绪');
  assert.equal(h.get('job-elapsed').textContent,'00:00');
  assert.equal(h.get('progress-bar').style.width,'0%');
  assert.equal(h.get('download-result').classList.contains('hidden'),true);
  assert.equal(h.get('download-result').href,'#');
  assert.equal(h.get('execution-progress').children.length,0);
  assert.equal(h.get('timing-progress').children.length,0);
  assert.equal(h.get('job-log').textContent,'尚无运行记录');
  assert.equal(h.saved.get('zest.parallel.dataset'),'new');
});

test('failed import preserves the current dataset and its records',async()=>{
  const db={datasets:[dataset('old')],jobs:[job('old-fit','old')],importError:true};
  const h=harness(db);
  await h.run('selectDataset("old")');await h.run('refreshJobs()');
  await assert.rejects(h.run('importPath("bad.mat")'),/无法读取文件/);
  assert.equal(h.run('S.dataset.id'),'old');
  assert.deepEqual(h.json('S.jobs.map(j=>j.id)'),['old-fit']);
  assert.equal(h.get('dataset-select').value,'old');
});

test('file picker upload starts a new dataset group just like path import',async()=>{
  const db={datasets:[dataset('old')],jobs:[job('old-fit','old')],imported:dataset('uploaded')};
  const h=harness(db);
  await h.run('selectDataset("old")');await h.run('refreshJobs()');h.run('attachEvents()');
  h.get('file-upload').files=[new Blob(['sample'])];
  await h.get('file-upload').onchange();
  assert.equal(h.run('S.datasetRoot'),'uploaded');
  assert.deepEqual(h.json('S.jobs'),[]);
  assert.equal(h.get('dataset-select').options.length,2);
  assert.equal(h.get('file-upload').value,'');
});

test('preprocessing keeps the original, output and jobs within the current group',async()=>{
  const db={datasets:[dataset('old'),dataset('new')],jobs:[job('old-fit','old')],
    preprocess:job('prep','new',{kind:'preprocess',status:'queued'})};
  const h=harness(db);
  await h.run('refreshDatasets()');await h.run('selectDataset("new")');
  db.jobs.unshift(db.preprocess);await h.run('preprocess()');
  h.run('S.maps={T_raw:[[300]]};');
  db.datasets.unshift(dataset('new-child','new'));
  db.jobs[0]=job('prep','new',{kind:'preprocess',output_dataset_id:'new-child'});
  await h.run('S.nextHealth=Date.now()+100000;poll()');
  assert.equal(h.run('S.dataset.id'),'new-child');
  assert.deepEqual(h.json('S.maps'),{});
  assert.deepEqual(h.json('[...S.datasetIds].sort()'),['new','new-child']);
  assert.deepEqual(h.json('S.jobs.map(j=>j.id)'),['prep']);
  assert.equal(h.get('preprocess-history').children.length,2);
  db.jobs.unshift(job('new-fit','new-child'));await h.run('refreshJobs()');
  await h.run('selectDataset("new")');await h.run('refreshJobs()');
  assert.deepEqual(h.json('S.jobs.map(j=>j.id)'),['new-fit','prep']);
});

test('page reload and reconnect restore only the selected import group',async()=>{
  const db={datasets:[dataset('new-child','new'),dataset('new'),dataset('old')],
    jobs:[job('old-fit','old'),job('new-fit','new-child')]};
  const h=harness(db);h.saved.set('zest.parallel.dataset','new-child');
  await h.run('loadWorkbench()');
  assert.equal(h.run('S.datasetRoot'),'new');
  assert.deepEqual(h.json('S.jobs.map(j=>j.id)'),['new-fit']);
  await h.run('S.online=false;S.nextReconnect=0;poll()');
  assert.equal(h.run('S.datasetRoot'),'new');
  assert.deepEqual(h.json('S.jobs.map(j=>j.id)'),['new-fit']);
});

test('an old active task stays locked but cannot restore old data on completion',async()=>{
  const db={datasets:[dataset('new'),dataset('old')],jobs:[job('old-prep','old',{kind:'preprocess',status:'running'})]};
  const h=harness(db);await h.run('selectDataset("new")');await h.run('refreshJobs()');
  assert.deepEqual(h.json('S.jobs'),[]);
  assert.equal(h.run('S.running'),'old-prep');assert.equal(h.get('run-fit').disabled,true);
  db.jobs[0]=job('old-prep','old',{kind:'preprocess',output_dataset_id:'old-child'});
  await h.run('S.nextHealth=Date.now()+100000;poll()');
  assert.equal(h.run('S.dataset.id'),'new');assert.equal(h.run('S.running'),null);
  assert.equal(h.get('run-fit').disabled,false);
  assert.ok(!db.calls.some(([url])=>url==='/api/datasets/old-child'));
});

test('a delayed job response from an earlier group cannot repopulate the new workspace',async()=>{
  const db={datasets:[dataset('old'),dataset('new')],jobs:[job('old-fit','old')]};
  let resolveJob;
  const h=harness(db,url=>url==='/api/jobs/old-fit'?new Promise(resolve=>{resolveJob=resolve;}):undefined);
  await h.run('selectDataset("old")');const pending=h.run('selectJob("old-fit")');
  await h.run('selectDataset("new")');resolveJob(response(db.jobs[0]));await pending;
  assert.equal(h.run('S.dataset.id'),'new');assert.equal(h.run('S.job'),null);
  assert.deepEqual(h.json('S.maps'),{});
});

test('a delayed result map from an earlier group cannot repopulate the new workspace',async()=>{
  const db={datasets:[dataset('old'),dataset('new')],jobs:[job('old-fit','old')]};
  let resolveMaps,announceMaps;const mapsRequested=new Promise(resolve=>{announceMaps=resolve;});
  const h=harness(db,url=>url==='/api/jobs/old-fit/maps'?new Promise(resolve=>{resolveMaps=resolve;announceMaps();}):undefined);
  await h.run('selectDataset("old")');const pending=h.run('selectJob("old-fit")');await mapsRequested;
  await h.run('selectDataset("new")');resolveMaps(response({T_raw:[[300]]}));await pending;
  assert.equal(h.run('S.dataset.id'),'new');assert.equal(h.run('S.job'),null);
  assert.deepEqual(h.json('S.maps'),{});
});

test('the last task selection wins when an earlier status response arrives late in the same group',async()=>{
  const db={datasets:[dataset('data')],jobs:[job('A','data'),job('B','data')]};
  let resolveA;
  const h=harness(db,url=>url==='/api/jobs/A'?new Promise(resolve=>{resolveA=resolve;}):undefined);
  await h.run('selectDataset("data")');
  const pending=h.run('selectJob("A")');await h.run('selectJob("B")');
  resolveA(response(db.jobs[0]));await pending;
  assert.equal(h.run('S.job.id'),'B');
  assert.ok(!db.calls.some(([url])=>url==='/api/jobs/A/maps'));
});

test('late maps from an earlier task cannot replace the newly selected task maps',async()=>{
  const db={datasets:[dataset('data')],jobs:[job('A','data'),job('B','data')]};
  let resolveMaps,announceMaps;const requested=new Promise(resolve=>{announceMaps=resolve;});
  const h=harness(db,url=>url==='/api/jobs/A/maps'?new Promise(resolve=>{resolveMaps=resolve;announceMaps();}):undefined);
  await h.run('selectDataset("data")');
  const pending=h.run('selectJob("A")');await requested;await h.run('selectJob("B")');
  const expected=h.json('S.maps');resolveMaps(response({T_raw:[[123]]}));await pending;
  assert.equal(h.run('S.job.id'),'B');assert.deepEqual(h.json('S.maps'),expected);
});

test('a newer task cancels the dataset load started by an earlier task selection',async()=>{
  const db={datasets:[dataset('original'),dataset('child','original')],jobs:[
    job('prep','original',{kind:'preprocess',output_dataset_id:'child'}),job('fit','original')]};
  let resolveData,announceData;const requested=new Promise(resolve=>{announceData=resolve;});
  const h=harness(db,url=>url==='/api/datasets/child'?new Promise(resolve=>{resolveData=resolve;announceData();}):undefined);
  await h.run('refreshDatasets()');await h.run('selectDataset("original")');
  const pending=h.run('selectJob("prep")');await requested;await h.run('selectJob("fit")');
  resolveData(response(db.datasets[1]));await pending;
  assert.equal(h.run('S.dataset.id'),'original');assert.equal(h.run('S.job.id'),'fit');
  assert.equal(h.saved.get('zest.parallel.dataset'),'original');
});

test('manual dataset selection invalidates a pending task selection within the same group',async()=>{
  const db={datasets:[dataset('original'),dataset('child','original')],jobs:[job('fit','original')]};
  let resolveJob;
  const h=harness(db,url=>url==='/api/jobs/fit'?new Promise(resolve=>{resolveJob=resolve;}):undefined);
  await h.run('refreshDatasets()');await h.run('selectDataset("original")');
  const pending=h.run('selectJob("fit")');await h.run('selectDataset("child")');
  resolveJob(response(db.jobs[0]));await pending;
  assert.equal(h.run('S.dataset.id'),'child');assert.equal(h.run('S.job'),null);
});

test('selecting a task can still load its processed dataset and results as one selection',async()=>{
  const db={datasets:[dataset('original'),dataset('child','original')],jobs:[job('fit','child')]};
  const h=harness(db);
  await h.run('refreshDatasets()');await h.run('selectDataset("original")');
  await h.run('selectJob("fit")');
  assert.equal(h.run('S.dataset.id'),'child');assert.equal(h.run('S.job.id'),'fit');
  assert.equal(h.run('S.datasetRoot'),'original');
  assert.deepEqual(h.json('S.maps.R2'),[[.99,.99,.99],[.99,.99,.99]]);
});

test('phase timings start collapsed and preserve disclosure state and focus target through updates',()=>{
  const h=harness({datasets:[],jobs:[]});
  h.run('renderTimings({id:"fit",status:"running",timings:{wall_seconds:10,phases:[{key:"fit",label:"拟合",seconds:10}],active_phase:"fit"}})');
  const box=h.get('timing-progress'),details=box.querySelector('details'),summary=details.querySelector('summary');
  assert.equal(box.classList.contains('hidden'),false);
  assert.equal(details.open,false);
  assert.equal(summary.textContent,'阶段计时 · 00:10');
  details.open=true;
  h.run('renderTimings({id:"fit",status:"running",timings:{wall_seconds:75,phases:[{key:"fit",label:"拟合",seconds:74}],other_seconds:1,active_phase:"fit"}})');
  assert.equal(box.querySelector('details'),details);
  assert.equal(details.querySelector('summary'),summary);
  assert.equal(details.open,true);
  assert.equal(summary.textContent,'阶段计时 · 01:15');
  const rows=details.querySelector('div').children;
  assert.equal(rows[0].children[1].textContent,'01:14');
  assert.equal(rows[0].classList.contains('active'),true);
  assert.equal(rows[1].children[0].textContent,'其他调度与管理');
  details.open=false;
  h.run('renderTimings({id:"fit",status:"completed",timings:{wall_seconds:80,phases:[{key:"fit",label:"拟合",seconds:80}],active_phase:"fit"}})');
  assert.equal(details.open,false);
  assert.equal(summary.textContent,'阶段计时 · 01:20');
  assert.equal(details.querySelector('div').children[0].classList.contains('active'),false);
});

test('switching tasks or clearing the workspace resets phase timing disclosure',()=>{
  const h=harness({datasets:[],jobs:[]}),box=h.get('timing-progress');
  h.run('renderTimings({id:"first",timings:{wall_seconds:1}})');
  box.querySelector('details').open=true;
  h.run('renderTimings({id:"second",timings:{wall_seconds:2}})');
  assert.equal(box.querySelector('details').open,false);
  assert.equal(box.querySelector('summary').textContent,'阶段计时 · 00:02');
  h.run('renderTimings({id:"legacy"})');
  assert.equal(box.classList.contains('hidden'),true);
  assert.equal(box.children.length,0);
  h.run('renderTimings({id:"second",timings:{wall_seconds:2}})');
  box.querySelector('details').open=true;
  h.run('resetJobPanel()');
  assert.equal(box.classList.contains('hidden'),true);
  assert.equal(box.children.length,0);
});

test('balanced preview uses 3n minus 4 and supports more blocks than columns',()=>{
  const h=harness({datasets:[],jobs:[]});
  h.run('S.dataset={pixels:119,shape:[7,17,20]};$("scope").value="all";updateScope()');
  assert.match(h.get('run-estimate').textContent,/18 进程 · 50 块 · 2–3 像素/);
  h.get('partition-mode').value='horizontal';h.run('updateScope()');
  assert.match(h.get('run-estimate').textContent,/7 块/);
  h.get('partition-mode').value='vertical';h.run('updateScope()');
  assert.match(h.get('run-estimate').textContent,/17 块/);
  h.get('partition-mode').value='grid';h.get('grid-rows').value='3';h.get('grid-columns').value='5';h.run('updateScope()');
  assert.match(h.get('run-estimate').textContent,/15 块 · 3 行 × 5 列/);
  assert.equal(h.get('stripe-count').disabled,true);
  assert.deepEqual(h.json('parallelOptions()'),{parallel_workers:18,stripe_count:0,partition_mode:'grid',grid_rows:3,grid_columns:5});
  h.get('grid-rows').value='101';h.get('grid-columns').value='100';h.run('updateScope()');
  assert.match(h.get('run-estimate').textContent,/10000/);
  h.get('fit-stage').value='stage2';h.run('updateScope()');
  assert.equal(h.get('grid-rows').disabled,true);
  assert.doesNotThrow(()=>h.run('parallelOptions()'));
});

test('hardware refresh updates automatic defaults, preserves manual values and never changes a running job',async()=>{
  const db={datasets:[],jobs:[]};let probes=0;
  const h=harness(db,url=>{if(url==='/api/hardware/refresh'){probes++;return response({...config,logical_cpus:8,parallel_workers:6,hardware:{cpu_name:'Fresh CPU',logical_cpus:8,memory_bytes:16*1024**3,detected_at:'2026-10-12T00:00:00Z',gpu:false}});}});
  h.run('attachEvents();S.dataset={id:"data",shape:[7,17,20],pixels:119};$("scope").value="all";S.running="job1";S.job={id:"job1",request:{parallel_workers:18}};');
  await h.run('refreshHardware()');
  assert.equal(Number(h.get('parallel-workers').value),6);
  assert.equal(h.get('device-name').textContent,'Fresh CPU');
  assert.match(h.get('device-detail').textContent,/8 线程/);
  assert.equal(h.run('S.job.request.parallel_workers'),18);
  h.get('parallel-workers').value='3';h.get('parallel-workers').oninput();
  await h.run('refreshHardware()');
  assert.equal(Number(h.get('parallel-workers').value),3);
  assert.match(h.get('notice').querySelector('span').textContent,/保留手动/);
  assert.equal(h.get('refresh-hardware').disabled,false);
  h.get('recommended-workers').onclick();
  assert.equal(Number(h.get('parallel-workers').value),6);
  assert.equal(JSON.parse(h.saved.get('zest.parallel.scheduling')).worker_mode,'auto');
  assert.equal(probes,2);
});

test('automatic preferences follow startup detection while manual grid choices survive reload',async()=>{
  const h=harness({datasets:[],jobs:[]});
  h.saved.set('zest.parallel.scheduling',JSON.stringify({worker_mode:'auto',parallel_workers:6,stripe_count:0,partition_mode:'grid',grid_rows:3,grid_columns:5}));
  await h.run('loadWorkbench()');
  assert.equal(Number(h.get('parallel-workers').value),18);
  assert.equal(Number(h.get('grid-rows').value),3);
  assert.equal(Number(h.get('grid-columns').value),5);
  h.saved.set('zest.parallel.scheduling',JSON.stringify({parallel_workers:4,stripe_count:6}));
  await h.run('loadWorkbench()');
  assert.equal(Number(h.get('parallel-workers').value),4);
  assert.equal(h.get('partition-mode').value,'vertical');
});

test('failed hardware refresh restores its button without erasing the previous inventory',async()=>{
  const h=harness({datasets:[],jobs:[]},url=>{if(url==='/api/hardware/refresh')return response({detail:'probe failed'},500);});
  await assert.rejects(h.run('refreshHardware()'),/probe failed/);
  assert.equal(h.run('S.config.logical_cpus'),20);
  assert.equal(h.run('S.refreshingHardware'),false);
  assert.equal(h.get('refresh-hardware').disabled,false);
});

test('heatmap renders exact block edges including horizontal partial-column boundaries',()=>{
  const db={datasets:[],jobs:[]},h=harness(db);
  h.run('S.dataset={id:"data"};S.preview={intensity:[[1,2],[3,4]]};S.job={dataset_id:"data",execution:{stripes:[{id:1},{id:2}],partition_boundaries:[[.5,1.5,1.5,1.5],[1.5,.5,1.5,1.5]]}};renderHeatmap()');
  assert.deepEqual(JSON.parse(JSON.stringify(db.plots.heatmap.layout.shapes)).map(({x0,y0,x1,y1})=>[x0,y0,x1,y1]),[[.5,1.5,1.5,1.5],[1.5,.5,1.5,1.5]]);
});
