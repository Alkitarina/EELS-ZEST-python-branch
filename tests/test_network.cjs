// Run with: node --test tests/test_network.cjs (no browser/packages required).
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../static/app.js'), 'utf8');

function harness(fetch) {
  const nodes = new Map();
  function element() {
    const classes = new Set();
    return {textContent:'', value:'', disabled:false, style:{}, options:[{}, {}],
      classList:{add:x=>classes.add(x),remove:x=>classes.delete(x),contains:x=>classes.has(x),
        toggle:(x,on)=>on?classes.add(x):classes.delete(x)},
      append(){}, replaceChildren(){}, setAttribute(){},
      querySelector(){return this.child ||= element();}};
  }
  const get = id => {if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
  const ctx = vm.createContext({fetch,AbortController,setTimeout,clearTimeout,console,
    document:{getElementById:get,createElement:element,createTextNode:x=>x,addEventListener(){}},
    localStorage:{getItem:()=>null,setItem(){}},structuredClone});
  vm.runInContext(source,ctx);
  return {ctx,get,run:code=>vm.runInContext(code,ctx)};
}
const response = (payload,status=200) => ({ok:status>=200&&status<300,status,json:async()=>payload});
const config = {parallel_workers:12,max_parallel_workers:61,partition_modes:['balanced','vertical','horizontal','grid'],hardware:{gpu:false},
  defaults:{crop_range:[-100,100],zlp_model_type:'pearson',stage2_T_grid:[100,300]}};

test('network failure locks submissions, explains uncertain POST, and never retries it',async()=>{
  let calls=0;
  const h=harness(async()=>{calls++;throw new TypeError('Failed to fetch');});
  h.run('S.initialized=true; S.dataset={id:"data"}; S.online=true;');
  await assert.rejects(h.run('api("/api/fit",{dataset_id:"data"})'),/是否提交成功尚未确认/);
  assert.equal(calls,1);
  assert.equal(h.run('S.online'),false);
  assert.equal(h.get('run-fit').disabled,true);
  assert.equal(h.get('file-upload').disabled,true);
  assert.equal(h.get('connection').classList.contains('offline'),true);
});

test('HTTP validation errors remain distinct from a lost connection',async()=>{
  const h=harness(async()=>response({detail:'像素坐标超出数据范围'},400));
  await assert.rejects(h.run('api("/api/fit",{})'),/像素坐标超出数据范围/);
  assert.equal(h.run('S.online'),true);
});

test('failed initial load can recover without a browser refresh',async()=>{
  let alive=false;
  const h=harness(async url=>{
    if(!alive)throw new TypeError('Failed to fetch');
    return response(url==='/api/config'?config:[]);
  });
  await h.run('poll()');
  assert.equal(h.run('S.initialized'),false);
  assert.equal(h.run('S.polling'),false);
  alive=true;
  await h.run('S.nextReconnect=0; poll()');
  assert.equal(h.run('S.initialized'),true);
  assert.equal(h.run('S.online'),true);
  assert.equal(h.get('import-path').disabled,false);
  assert.match(h.get('notice').querySelector().textContent,/连接已恢复/);
});

test('idle page probes health and detects a backend exit',async()=>{
  const calls=[];
  const h=harness(async url=>{calls.push(url);throw new TypeError('Failed to fetch');});
  await h.run('S.initialized=true; S.online=true; poll()');
  assert.deepEqual(calls,['/api/health']);
  assert.equal(h.run('S.online'),false);
  assert.match(h.get('notice').querySelector().textContent,/start_web.ps1/);
});

test('reconnect refreshes interrupted job state without resubmitting work',async()=>{
  const calls=[];
  const job={id:'job1',kind:'fit',status:'interrupted',dataset_id:'data',request:{stage:'both'},created_at:1};
  const h=harness(async (url,options)=>{calls.push([url,options.method]);return response(url==='/api/jobs'?[job]:url==='/api/config'?config:{status:'ok'});});
  // Keep the test focused on network/state; chart rendering is independent.
  h.run('selectJob=async id=>{S.job=S.jobs.find(j=>j.id===id);}; S.datasetRoot="data"; S.datasetIds.add("data"); S.initialized=true; S.online=false; S.running="job1"; S.job={id:"job1"};');
  await h.run('poll()');
  assert.equal(h.run('S.running'),null);
  assert.equal(h.run('S.job.status'),'interrupted');
  assert.equal(h.run('S.online'),true);
  assert.match(h.get('notice').querySelector().textContent,/不会自动续跑/);
  assert.deepEqual(calls.map(x=>x[0]),['/api/health','/api/config','/api/jobs']);
  assert.ok(calls.every(x=>x[1]===undefined));
});

test('hung health request times out and locks disconnected actions',async()=>{
  const h=harness(async (url,{signal})=>new Promise((resolve,reject)=>{
    signal.addEventListener('abort',()=>reject(new Error('aborted')),{once:true});
  }));
  await assert.rejects(h.run('requestJSON("/api/health",{},5)'),/响应超时/);
  assert.equal(h.run('S.online'),false);
  assert.equal(h.get('cancel-job').disabled,true);
});

test('a connection loss while reading a response is also reported clearly',async()=>{
  const h=harness(async()=>({ok:true,status:200,json:async()=>{throw new TypeError('terminated');}}));
  await assert.rejects(h.run('api("/api/jobs")'),/无法连接本机后端/);
  assert.equal(h.run('S.online'),false);
});
