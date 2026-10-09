'use strict';
const $ = id => document.getElementById(id);
const S = {config:null,dataset:null,datasets:[],datasetRoot:null,datasetIds:new Set(),scopeRevision:0,selectionRevision:0,preview:null,pixel:[0,0],result:null,maps:{},job:null,running:null,jobs:[],view:'spectrum',scale:'linear',loading:false,polling:false,online:null,initialized:false,nextReconnect:0,nextHealth:0,handled:new Set()};
const NAMES = {intensity:'积分强度',T_raw:'原始拟合温度 / K',T_valid:'数据有效温度 / K',T_joint:'联合温度 / K',T_uncertainty:'温度扫描谷宽 / K',T_joint_uncertainty:'联合温度谷宽 / K',T_inferred_mask:'空间推断像素',T_valid_mask:'有效温度像素',R2:'拟合 R²',NRMSE:'NRMSE',reliability:'可靠度',attention:'Attention 选中比例',cluster:'ZLP 聚类',membership:'聚类归属度',interface:'ZLP 界面分数',template_deviation:'模板偏离',fit_status:'拟合状态',T_joint_shift:'联合温度变化 / K'};
const STATUS = {queued:'排队中',running:'计算中',completed:'已完成',completed_with_errors:'部分失败',failed:'失败',cancelled:'已停止',interrupted:'已中断'};
const ACTIVE = ['queued','running'];
const valid = x => x!==null && x!==undefined && Number.isFinite(Number(x));
const fmt = (x,n=2) => valid(x) ? Number(x).toLocaleString('en-US',{maximumFractionDigits:n}) : '—';
const duration = x => {x=Math.floor(x||0);return x>=3600?`${Math.floor(x/3600)}:${String(Math.floor(x/60)%60).padStart(2,'0')}:${String(x%60).padStart(2,'0')}`:`${String(Math.floor(x/60)).padStart(2,'0')}:${String(x%60).padStart(2,'0')}`;};
const opt = (value,label) => {const e=document.createElement('option');e.value=value;e.textContent=label;return e;};
function tell(message,error=false){$('notice').classList.remove('hidden');$('notice').classList.toggle('error',error);$('notice').querySelector('span').textContent=message;}
function setConnection(online){
  S.online=online;$('connection').classList.toggle('offline',!online);$('connection').replaceChildren();
  const dot=document.createElement('i');$('connection').append(dot,document.createTextNode(online?'本地服务在线':'后端连接中断 · 自动重连中'));
  if(!online&&S.running){$('job-message').textContent='后端连接中断，计算状态待确认；下方保留最后收到的进度。';$('progress-bar').classList.remove('indeterminate');}
  busy(S.loading);
}
async function requestJSON(url,options={},timeoutMs=15000){
  const controller=new AbortController();const timer=timeoutMs>0?setTimeout(()=>controller.abort(),timeoutMs):null;
  let response,payload;
  try{
    try{
      response=await fetch(url,{...options,signal:controller.signal});
      payload=await response.json();
    }catch(cause){
      // HTTP errors are handled below; a rejected fetch/body read is a lost
      // connection, not evidence that a fit or its scientific data is invalid.
      if(cause instanceof SyntaxError){setConnection(true);throw new Error(`后端响应格式异常 (${response?.status||'未知状态'})，请查看 runtime/logs。`);}
      setConnection(false);S.nextReconnect=Date.now()+5000;
      let message=controller.signal.aborted?'后端响应超时。':'无法连接本机后端。';
      message+='请重新运行 start_web.ps1；页面会自动重连。已保存的像素仍保留，未完成计算不会自动续跑。';
      if(options.method&&options.method!=='GET')message+=' 本次操作是否提交成功尚未确认，重连后请先查看计算记录。';
      const error=new Error(message);error.connectionError=true;throw error;
    }
    setConnection(true);
    if(!response.ok)throw new Error(typeof payload?.detail==='string'?payload.detail:JSON.stringify(payload?.detail||payload||`请求失败 (${response.status})`));
    return payload;
  }finally{if(timer!==null)clearTimeout(timer);}
}
async function api(url,body,timeoutMs){
  // Never retry POST automatically: the server may already have accepted it.
  return requestJSON(url,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)},timeoutMs??(body===undefined?15000:0));
}
async function action(fn){try{await fn();}catch(error){tell(error.message,true);}}
function busy(loading){S.loading=loading;const unavailable=S.online!==true||!S.initialized;const oldBackend=S.config&&S.config.max_parallel_workers===undefined;const locked=unavailable||loading||!!S.running||oldBackend;for(const id of ['run-fit','preprocess'])$(id).disabled=locked||!S.dataset;for(const id of ['import-path','load-reference','file-upload','dataset-select'])$(id).disabled=unavailable||loading;$('cancel-job').disabled=unavailable;}
function step(id){$(id).scrollIntoView({behavior:'smooth',block:'start'});}

function datasetRoot(id,entries=S.datasets){
  const byId=new Map(entries.map(d=>[d.id,d])),seen=new Set();
  while(byId.get(id)?.parent_id&&!seen.has(id)){seen.add(id);id=byId.get(id).parent_id;}
  return id;
}
function syncDatasetOptions(){
  // Each import has its own ID, even when the filename is unchanged. Only
  // descendants of that import belong in this workspace's history.
  const entries=S.datasets.filter(d=>S.datasetRoot&&datasetRoot(d.id)===S.datasetRoot);
  S.datasetIds=new Set(entries.map(d=>d.id));
  $('dataset-select').replaceChildren(opt('','选择已载入数据…'),...entries.map(d=>opt(d.id,d.name)));
  $('dataset-select').value=S.dataset?.id||'';
}
function jobInScope(j){return S.datasetIds.has(j.dataset_id)||S.datasetIds.has(j.output_dataset_id);}
function resetJobPanel(){
  $('job-status').textContent='就绪';$('job-status').className='pill';
  $('job-message').textContent='选择数据和计算范围后即可开始。';$('job-count').textContent='—';$('job-elapsed').textContent='00:00';
  $('progress-bar').classList.remove('indeterminate');$('progress-bar').style.width='0%';
  $('cancel-job').classList.add('hidden');$('download-result').classList.add('hidden');$('download-result').href='#';
  $('job-log').textContent='尚无运行记录';$('job-log').closest('details').open=false;
  for(const id of ['execution-progress','timing-progress']){$(id).replaceChildren();$(id).classList.add('hidden');}
}
function resetDatasetWorkspace(){
  S.scopeRevision++;S.job=null;S.result=null;S.maps={};S.preview=null;S.jobs=[];S.pixel=[0,0];S.handled.clear();
  S.view='spectrum';for(const b of $('plot-tabs').children)b.classList.toggle('active',b.dataset.view===S.view);
  if($('fit-stage').value==='stage2')$('fit-stage').value='both';
  $('preprocess-history').replaceChildren();$('notice').classList.add('hidden');
  resetJobPanel();renderJobs();renderMapOptions();renderPixelStats();Plotly.purge('heatmap');$('map-empty').classList.remove('hidden');renderSpectrum();
}
async function refreshDatasets(){
  const entries=await api('/api/datasets');S.datasets=entries;
  if(S.dataset&&!entries.some(d=>d.id===S.dataset.id))S.datasets=[S.dataset,...entries];
  syncDatasetOptions();return S.datasets;
}
async function selectDataset(id,imported=null){
  if(!id)return;const selection=++S.selectionRevision;const d=imported||await api(`/api/datasets/${id}`);
  if(selection!==S.selectionRevision)return;
  S.datasets=[d,...S.datasets.filter(entry=>entry.id!==id)];const root=datasetRoot(id);
  if(root!==S.datasetRoot){S.datasetRoot=root;resetDatasetWorkspace();}
  if(S.dataset?.id!==id){S.result=null;S.maps={};renderMapOptions();}
  S.dataset=d;localStorage.setItem('zest.parallel.dataset',id);syncDatasetOptions();
  if(S.job&&S.job.dataset_id!==id&&S.job.output_dataset_id!==id){S.result=null;S.maps={};S.job=null;resetJobPanel();renderMapOptions();}
  S.preview=null;
  S.pixel=[Math.min(S.pixel[0],S.dataset.shape[0]-1),Math.min(S.pixel[1],S.dataset.shape[1]-1)];
  $('pixel-y').max=S.dataset.shape[0];$('pixel-x').max=S.dataset.shape[1];
  for(const id of ['roi-y0','roi-y1'])$(id).max=S.dataset.shape[0];
  for(const id of ['roi-x0','roi-x1'])$(id).max=S.dataset.shape[1];
  $('roi-y1').value=Math.min(Number($('roi-y1').value),S.dataset.shape[0]);$('roi-x1').value=Math.min(Number($('roi-x1').value),S.dataset.shape[1]);
  $('pixel-count').textContent=fmt(S.dataset.pixels,0);$('spatial-shape').textContent=`${S.dataset.shape[0]} × ${S.dataset.shape[1]} pixels`;
  $('channel-count').textContent=fmt(S.dataset.shape[2],0);$('energy-step').textContent=`${fmt(S.dataset.energy_step,4)} meV / channel`;
  $('energy-range').textContent=`${fmt(S.dataset.energy_min,1)} → ${fmt(S.dataset.energy_max,1)}`;
  $('dataset-pill').textContent=S.dataset.history.length?'已预处理':'原始数据';$('dataset-pill').classList.add('ready');
  $('file-info').textContent=S.dataset.name;$('file-info').title=S.dataset.source;
  $('preprocess-history').replaceChildren();
  for(const h of S.dataset.history){for(const text of [h.zlp_alignment?'ZLP 已对齐':null,h.svd_rank?`SVD rank ${h.svd_rank}`:null,h.smoothing_points>1?`平滑 ${h.smoothing_points}`:null])if(text){const chip=document.createElement('span');chip.textContent=text;$('preprocess-history').append(chip);}}
  await refreshDatasets();if(selection!==S.selectionRevision)return;
  S.jobs=S.jobs.filter(jobInScope);renderJobs();await selectPixel(...S.pixel);if(selection!==S.selectionRevision)return;updateScope();busy(S.loading);
}
async function selectPixel(y,x){
  if(!S.dataset)return;y=Math.trunc(y);x=Math.trunc(x);
  if(!(y>=0&&y<S.dataset.shape[0]&&x>=0&&x<S.dataset.shape[1]))throw new Error('像素坐标超出数据范围');
  S.pixel=[y,x];$('pixel-y').value=y+1;$('pixel-x').value=x+1;$('selected-label').textContent=`Y ${String(y+1).padStart(2,'0')} / X ${String(x+1).padStart(2,'0')}`;
  const id=S.dataset.id;const preview=await api(`/api/datasets/${id}/preview?y=${y}&x=${x}`);
  if(id!==S.dataset?.id||y!==S.pixel[0]||x!==S.pixel[1])return;S.preview=preview;
  await loadPixelResult();renderHeatmap();renderSpectrum();updateScope();
}
async function loadPixelResult(){
  S.result=null;if(!S.job||S.job.kind!=='fit'||S.job.dataset_id!==S.dataset?.id){renderPixelStats();return;}
  const [y,x]=S.pixel;const uid=S.job.id;const r=await api(`/api/jobs/${uid}/pixel?y=${y}&x=${x}`);
  if(uid!==S.job?.id||y!==S.pixel[0]||x!==S.pixel[1])return;
  S.result=r.available?r:null;renderPixelStats();
}
function renderPixelStats(){
  const r=S.result;for(const id of ['pixel-temp','pixel-unc','pixel-r2','pixel-rel'])$(id).textContent='—';
  if(!r){$('pixel-result-note').textContent='选中像素尚无拟合结果';return;}
  if(r.error){$('pixel-result-note').textContent=`计算失败：${r.error}`;return;}
  $('pixel-temp').textContent=valid(r.T)?`${fmt(r.T,2)} K`:'—';$('pixel-unc').textContent=valid(r.T_uncertainty)?`± ${fmt(r.T_uncertainty,2)} K`:'—';
  $('pixel-r2').textContent=valid(r.metrics?.R2)?r.metrics.R2.toFixed(7):'—';$('pixel-rel').textContent=fmt(r.reliability,3);
  const [y,x]=S.pixel;const inferred=S.maps.T_inferred_mask?.[y]?.[x];const approved=S.maps.T_valid_mask?.[y]?.[x];
  $('pixel-result-note').textContent=r.stage2_error?`Stage 2 失败：${r.stage2_error}`:r.stage2_status==='not_run'?'Stage 1 已完成，可继续 Stage 2':inferred?'此像素在联合图中为空间推断；上方数值是独立拟合结果。':approved?'通过原版数据有效性判定':'未通过数据有效性判定，请检查温度扫描曲线。';
}

const commonLayout={font:{family:'Segoe UI, Microsoft YaHei, sans-serif',size:10,color:'#64748b'},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(255,255,255,.35)',hoverlabel:{bgcolor:'#fff',font:{size:11}},margin:{t:16,r:21,b:47,l:66},xaxis:{title:{font:{size:10}},gridcolor:'#f1f5f9',zerolinecolor:'#dbeafe',linecolor:'#e2e8f0',tickfont:{size:10}},yaxis:{title:{font:{size:10}},gridcolor:'#f1f5f9',zerolinecolor:'#dbeafe',linecolor:'#e2e8f0',tickfont:{size:10}},legend:{orientation:'h',y:1.10,x:0,font:{size:10},itemwidth:25},showlegend:true};
const chartConfig={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d','sendChartToCloud'],toImageButtonOptions:{format:'png',scale:2,filename:'EELS-ZEST'}};
function baseLayout(){return structuredClone(commonLayout);}
function renderMapOptions(){const prior=$('map-select').value;$('map-select').replaceChildren(...['intensity',...Object.keys(S.maps).filter(k=>NAMES[k])].map(k=>opt(k,NAMES[k])));if([...$('map-select').options].some(o=>o.value===prior))$('map-select').value=prior;}
function renderHeatmap(){
  if(!S.preview)return;$('map-empty').classList.add('hidden');const choice=$('map-select').value;const source=choice==='intensity'?S.preview.intensity:S.maps[choice];if(!source)return;const z=source.map(row=>row.map(v=>typeof v==='boolean'?Number(v):v));
  const ny=z.length,nx=z[0].length;const layout=baseLayout();layout.margin={t:13,r:48,b:34,l:41};layout.showlegend=false;layout.plot_bgcolor='rgba(255,255,255,.20)';layout.xaxis={...layout.xaxis,title:{text:'X / pixel',font:{size:9}},constrain:'domain',dtick:nx<=8?1:undefined};layout.yaxis={...layout.yaxis,title:{text:'Y / pixel',font:{size:9}},autorange:'reversed',scaleanchor:'x',constrain:'domain',dtick:ny<=8?1:undefined};
  layout.uirevision=`${S.dataset.id}-${choice}`;
  if(S.job?.dataset_id===S.dataset.id && S.job.execution?.stripes?.length>1){
    const rows=S.job.coords.map(p=>p[0]+1);const y0=Math.min(...rows)-.5,y1=Math.max(...rows)+.5;
    layout.shapes=S.job.execution.stripes.slice(0,-1).map(st=>({type:'line',x0:st.columns.at(-1)+.5,x1:st.columns.at(-1)+.5,y0,y1,line:{color:'#f8fafc',width:1.5,dash:'dot'}}));
  }
  const scale=choice.startsWith('T_')&&choice!=='T_inferred_mask'&&choice!=='T_valid_mask'?'Turbo':[[0,'#10275a'],[.22,'#235eb0'],[.5,'#0ea5b5'],[.77,'#79d5ad'],[1,'#f6f5b0']];
  Plotly.react('heatmap',[{type:'heatmap',z,x:Array.from({length:nx},(_,i)=>i+1),y:Array.from({length:ny},(_,i)=>i+1),colorscale:scale,hoverongaps:false,colorbar:{thickness:8,len:.83,x:1.03,tickfont:{size:9},outlinewidth:0},hovertemplate:'Y %{y} · X %{x}<br>%{z:.5g}<extra></extra>'},{type:'scatter',x:[S.pixel[1]+1],y:[S.pixel[0]+1],mode:'markers',marker:{symbol:'square-open',color:'#fff',size:12,line:{width:2}},hoverinfo:'skip'}],layout,{...chartConfig,displayModeBar:false});
  const el=$('heatmap');if(el.removeAllListeners)el.removeAllListeners('plotly_click');el.on('plotly_click',ev=>{const p=ev.points[0];if(p?.data?.type==='heatmap')action(()=>selectPixel(p.y-1,p.x-1));});
  $('map-caption').textContent=choice==='T_joint'?'联合温度 · 有效数据锚点 + 原版单向空间推断':`${NAMES[choice]} · 点击选点 · 拖动缩放`;
}
function curve(x,y,name,color,width=1.5,extra={}){return {type:'scatter',mode:'lines',x,y,name,line:{color,width},hovertemplate:'%{x:.2f} meV<br>%{y:.5g}<extra>'+name+'</extra>',...extra};}
function renderSpectrum(){
  const layout=baseLayout();const r=S.result&&!S.result.error?S.result:null;const traces=[];let caption='';
  layout.xaxis.title.text='能量损失 / meV';layout.yaxis.title.text='归一化强度';layout.uirevision=`${S.dataset?.id}-${S.view}-${S.scale}`;layout.margin.t=33;
  if(S.view==='spectrum'){
    if(r){traces.push(curve(r.energy,r.raw,'实验谱','#94a3b8',1.5),curve(r.energy,r.model,'总拟合','#2563eb',1.6),curve(r.energy,r.zlp,'ZLP','#f59e0b',1.4),curve(r.energy,r.vibrational,'振动信号','#0d9488',1.5));caption='归一化实验谱、总拟合、ZLP 与振动分量';}
    else if(S.preview){const norm=Math.max(...S.preview.spectrum.map(v=>Math.abs(v||0)))||1;traces.push(curve(S.preview.energy,S.preview.spectrum.map(v=>v/norm),'实验谱','#2563eb'));caption='预处理状态见数据面板 · 当前像素尚未拟合';}
    layout.yaxis.type=S.scale;
  }else if(S.view==='residual'){
    layout.yaxis.title.text='实验谱 − 拟合谱';if(r){traces.push(curve(r.energy,r.residual,'残差','#64748b',1.2));caption=`R² ${fmt(r.metrics.R2,7)} · NRMSE ${fmt(r.metrics.NRMSE,7)}`;}
  }else if(S.view==='modes'){
    layout.yaxis.title.text='模态系数';layout.barmode='group';if(r){traces.push({type:'bar',x:r.EM,y:r.PM_stokes,name:'Stokes',marker:{color:'#2563eb'}},{type:'bar',x:r.EM,y:r.PM_antistokes,name:'Anti-Stokes',marker:{color:'#14b8a6'}});caption=`${r.EM.length} 个模态中心 · 系数对应当前阶段输出`;}
  }else if(S.view==='temperature'){
    layout.xaxis.title.text='温度 / K';layout.yaxis.title.text='剖面目标函数';if(r?.T_grid.length){traces.push({type:'scatter',mode:'lines+markers',x:r.T_grid,y:r.cost_grid,name:'原始目标',marker:{size:4},line:{color:'#94a3b8',width:1.2}},{type:'scatter',mode:'lines',x:r.T_grid,y:r.cost_smooth,name:'平滑目标',line:{color:'#2563eb',width:1.7}});layout.shapes=[{type:'line',x0:r.T,x1:r.T,y0:0,y1:1,yref:'paper',line:{color:'#0d9488',dash:'dot',width:1.5}}];caption=`拟合温度 ${fmt(r.T)} K · 扫描谷宽 ± ${fmt(r.T_uncertainty)} K（不是统计置信区间）`;}
  }else if(S.view==='attention'){
    layout.yaxis.title.text='特征 / 权重';layout.yaxis.range=[0,1.05];if(r?.attention.length){traces.push(curve(r.energy,r.attention,'特征分数','#7c3aed',1.4),curve(r.energy,r.alpha,'拟合 Alpha','#0d9488',1.4));caption='Stage 2 原版 Attention 与拟合窗口';}
  }
  if(!traces.length){layout.annotations=[{text:S.preview?'此视图将在对应阶段计算完成后显示':'载入数据后，在热图中选择一个像素',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false,font:{color:'#94a3b8',size:12}}];layout.xaxis.range=[-250,250];layout.yaxis.range=[0,1];caption='等待计算结果';}
  Plotly.react('spectrum',traces,layout,chartConfig);$('spectrum-caption').textContent=caption;
}

function updateScope(){
  const scope=$('scope').value;$('roi-fields').classList.toggle('hidden',scope!=='roi');
  let count=1,columns=1;
  if(scope==='all'){count=S.dataset?.pixels||0;columns=S.dataset?.shape[1]||1;}
  if(scope==='roi'){columns=Math.max(0,Number($('roi-x1').value)-Number($('roi-x0').value)+1);count=Math.max(0,Number($('roi-y1').value)-Number($('roi-y0').value)+1)*columns;}
  if($('fit-stage').value==='stage2' && S.job?.request?.stage==='stage1'){count=S.job.total_pixels;columns=new Set((S.job.coords||[]).map(p=>p[1])).size;}
  const parallel=$('execution-mode').value==='stripes';
  $('parallel-options').classList.toggle('hidden',!parallel);
  $('parallel-workers').disabled=!parallel||count<=1;
  $('stripe-count').disabled=!parallel||count<=1||$('fit-stage').value==='stage2';
  let options;try{options=parallelOptions();}catch(e){$('run-estimate').textContent=e.message;return;}
  const workers=Math.min(options.parallel_workers,count),stripes=Math.min(columns,options.stripe_count||3*options.parallel_workers);
  const detail=count<=1?'单像素直接计算':parallel?($('fit-stage').value==='stage2'?`${workers} 进程扫描温度`:`${workers} 进程 · ${stripes} 条带 · 首轮最多 ${Math.min(workers,stripes)} 条同时计算`):'串行对照 · 列优先热启动';
  $('run-estimate').textContent=`将计算 ${fmt(count,0)} 个像素 · ${detail}`;
  $('execution-note').textContent=parallel?'空闲进程自动领取下一条带；条内蛇形热启动，全图空间拟合保留跨条带邻域。条带数不超过所选列数。':'保留原程序的列优先遍历、热启动和数值库线程设置，供同一数据集对照。';
}
function parallelOptions(){
  const parallel_workers=$('parallel-workers').disabled?(S.config?.parallel_workers||12):Number($('parallel-workers').value),stripe_count=$('stripe-count').disabled?0:Number($('stripe-count').value);
  if(!Number.isInteger(parallel_workers)||parallel_workers<1||parallel_workers>(S.config?.max_parallel_workers||61))throw new Error('请输入有效的整数工作进程数');
  if(!Number.isInteger(stripe_count)||stripe_count<0||stripe_count>10000)throw new Error('条带数需为 0–10000 的整数，0 表示自动');
  return {parallel_workers,stripe_count};
}
function settings(){
  const low=Number($('temp-min').value),high=Number($('temp-max').value),step=Number($('temp-step').value);
  if(!Number.isFinite(step)||step<=0||high<=low||Math.floor((high-low)/step)>999)throw new Error('温度范围或步长无效');
  let advanced;try{advanced=JSON.parse($('advanced-json').value||'{}');}catch{throw new Error('高级参数不是有效的 JSON');}
  if(!advanced||Array.isArray(advanced)||typeof advanced!=='object')throw new Error('高级参数应为 JSON 对象');
  return {crop_range:[Number($('crop-min').value),Number($('crop-max').value)],zlp_model_type:$('model-type').value,stage1_speed_profile:$('profile').value,stage1_hierarchical_enabled:$('hierarchy').checked,datacube_stage1_auto_sparse_pm_enabled:$('calibration').checked,stage2_T_grid:Array.from({length:Math.floor((high-low)/step)+1},(_,i)=>low+i*step),...advanced};
}
async function startFit(){
  if(!S.dataset)return;busy(true);
  try{const stage=$('fit-stage').value;const job=await api('/api/fit',{dataset_id:S.dataset.id,device:$('backend').value,execution_mode:$('execution-mode').value,...parallelOptions(),scope:$('scope').value,pixel:S.pixel,roi:['roi-y0','roi-x0','roi-y1','roi-x1'].map(id=>Number($(id).value)-1),stage,source_job_id:stage==='stage2'?S.job?.id:null,settings:settings()});S.job=job;S.running=job.id;S.maps={};S.result=null;renderMapOptions();renderPixelStats();renderHeatmap();renderSpectrum();renderJob(job);await refreshJobs();step('progress-card');}finally{busy(false);}
}
async function preprocess(){if(!S.dataset)return;busy(true);try{const job=await api('/api/preprocess',{dataset_id:S.dataset.id,device:$('backend').value,align:$('align').checked,rank:Number($('svd-rank').value),smooth_points:Number($('smooth-points').value)});S.running=job.id;S.job=job;renderJob(job);await refreshJobs();}finally{busy(false);}}
function renderJob(j){
  const active=ACTIVE.includes(j.status);$('job-status').textContent=STATUS[j.status]||j.status;$('job-status').className=`pill ${active?'busy':j.status==='completed'?'ready':''}`;
  $('job-message').textContent=j.message;$('job-elapsed').textContent=duration(j.elapsed_seconds);$('job-count').textContent=j.kind==='fit'?`${fmt(j.completed_pixels||0,0)} / ${fmt(j.total_pixels||0,0)} 像素已保存`:'预处理整个数据集';
  $('progress-bar').classList.toggle('indeterminate',active&&(!j.done||j.phase==='hierarchy'));$('progress-bar').style.width=active?`${Math.min(100,100*j.done/Math.max(j.total,1))}%`:j.status.startsWith('completed')?'100%':`${100*(j.completed_pixels||0)/Math.max(j.total_pixels||1,1)}%`;
  $('cancel-job').classList.toggle('hidden',!active||j.phase==='export');$('download-result').classList.toggle('hidden',!j.download_ready);$('download-result').href=`/api/jobs/${j.id}/download`;
  $('job-log').textContent=(j.events||[]).map(e=>`${new Date(e.time*1000).toLocaleTimeString('zh-CN',{hour12:false})}  ${e.message}`).join('\n')||'任务已创建';
  renderExecution(j);renderTimings(j);
}
function executionLabel(j){return j.execution?.mode==='stripes'?'竖条并行':j.execution?.mode==='serial'?'串行对照':j.request?.execution_mode==='stripes'?'竖条并行':'串行对照';}
function renderExecution(j){
  const box=$('execution-progress'),wasOpen=box.querySelector('details')?.open;box.replaceChildren();box.classList.toggle('hidden',j.kind!=='fit');if(j.kind!=='fit')return;
  const execution=j.execution||{},stripes=execution.stripes||[];const heading=document.createElement('p');heading.className='execution-heading';heading.textContent=`${executionLabel(j)} · ${execution.workers||j.request?.parallel_workers||1} 个工作进程${stripes.length?` · ${stripes.length} 条带`:''}`;box.append(heading);
  const details=document.createElement('details');details.open=wasOpen??stripes.length<=12;
  const summary=document.createElement('summary');summary.textContent=`条带进度 · ${stripes.filter(s=>s.status==='completed').length}/${stripes.length} 已完成`;details.append(summary);
  const list=document.createElement('div');list.className='stripe-list';details.append(list);if(stripes.length)box.append(details);
  for(const st of execution.stripes||[]){
    const row=document.createElement('div');row.className='stripe-row';
    const title=document.createElement('div');title.className='stripe-title';
    const label=document.createElement('span');label.textContent=`条带 ${st.id} · X ${st.columns[0]}–${st.columns.at(-1)}`;
    const number=document.createElement('span');number.textContent=`${st.done}/${st.total}`;title.append(label,number);
    const track=document.createElement('div');track.className='stripe-track';const bar=document.createElement('div');bar.style.width=`${100*st.done/Math.max(1,st.total)}%`;track.append(bar);
    const detail=document.createElement('small');detail.textContent=st.status==='completed'?`Stage 1 完成 · PID ${st.pid} · ${duration(st.wall_seconds||st.elapsed_seconds)}`:`${st.status==='queued'?'等待领取':STATUS[st.status]||st.status}${st.pid?` · PID ${st.pid}`:''}${st.current_pixel?` · Y${st.current_pixel[0]} X${st.current_pixel[1]}`:''}`;
    row.append(title,track,detail);list.append(row);
  }
  const times=execution.phase_seconds||{};if(!j.timings&&Object.keys(times).length){const timing=document.createElement('p');timing.className='execution-timing';const labels={calibration:'标定',stage1:'Stage 1',hierarchy:'空间拟合',stage2:'Stage 2'};timing.textContent='历史阶段记录 · '+Object.entries(times).map(([k,v])=>`${labels[k]||k} ${duration(v)}`).join(' · ');box.append(timing);}
}
function renderTimings(j){
  const box=$('timing-progress'),t=j.timings;box.classList.toggle('hidden',!t);if(!t){box.replaceChildren();return;}
  let details=box.querySelector('details');
  if(!details||details.dataset.jobId!==j.id){
    details=document.createElement('details');details.className='timing-details';details.dataset.jobId=j.id;details.open=false;
    details.append(document.createElement('summary'),document.createElement('div'));box.replaceChildren(details);
  }
  // Keep the disclosure and its focus intact while polling updates the values.
  details.querySelector('summary').textContent=`阶段计时 · ${duration(t.wall_seconds)}`;
  const rows=details.querySelector('div');rows.replaceChildren();
  const phases=[...(t.phases||[])];if(t.other_seconds>0.001)phases.push({key:'other',label:'其他调度与管理',seconds:t.other_seconds});
  for(const phase of phases){
    const row=document.createElement('div');row.className='timing-row';row.classList.toggle('active',ACTIVE.includes(j.status)&&phase.key===t.active_phase);
    const label=document.createElement('span');label.textContent=phase.label;
    const value=document.createElement('b');value.textContent=phase.seconds<60?`${fmt(phase.seconds,2)} 秒`:duration(phase.seconds);
    row.append(label,value);rows.append(row);
  }
  const note=document.createElement('p');note.className='timing-note';note.textContent='各项时间不重复相加；计算项含调度与等待，像素保存单独计时。';rows.append(note);
}
async function refreshJobs(){
  const entries=await api('/api/jobs');S.jobs=entries.filter(jobInScope);
  // A hidden task from an older import still occupies the compute queue.
  // Keep the submission lock, but never restore its UI or select its output.
  S.running=entries.find(j=>ACTIVE.includes(j.status))?.id||null;renderJobs();busy(S.loading);return S.jobs;
}
function renderJobs(){
  $('jobs-list').replaceChildren();if(!S.jobs.length){const p=document.createElement('div');p.className='muted padded';p.textContent='尚无计算记录。先选择一个像素运行，检查谱线与温度扫描。';$('jobs-list').append(p);return;}
  for(const j of S.jobs){const row=document.createElement('div');row.className=`job-row ${S.job?.id===j.id?'selected':''}`;row.tabIndex=0;row.setAttribute('role','button');row.setAttribute('aria-label',`查看任务 ${j.id}`);const icon=document.createElement('div');icon.className='job-icon';icon.textContent=j.kind==='preprocess'?'≋':'∿';const main=document.createElement('div');main.className='job-main';const title=document.createElement('div');title.className='job-name';title.textContent=j.kind==='preprocess'?`ZLP / SVD 预处理 · ${j.device.toUpperCase()}`:`${j.request.stage==='stage1'?'Stage 1':j.request.stage==='stage2'?'Stage 2':'两阶段拟合'} · ${j.total_pixels||1} 像素 · ${executionLabel(j)}`;const meta=document.createElement('div');meta.className='job-meta';meta.textContent=`${new Date(j.created_at*1000).toLocaleString('zh-CN',{hour12:false})} · ${j.id}`;main.append(title,meta);const badge=document.createElement('span');badge.className=`pill ${ACTIVE.includes(j.status)?'busy':j.status==='completed'?'ready':''}`;badge.textContent=STATUS[j.status]||j.status;const t=document.createElement('span');t.className='job-time';t.textContent=duration(j.elapsed_seconds);row.append(icon,main,badge,t);row.onclick=()=>action(()=>selectJob(j.id));row.onkeydown=e=>{if(e.key==='Enter')row.click();};$('jobs-list').append(row);}
}
async function selectJob(id){
  const revision=S.scopeRevision,j=await api(`/api/jobs/${id}`);if(revision!==S.scopeRevision||!jobInScope(j))return;
  const did=j.output_dataset_id||j.dataset_id;if(did!==S.dataset?.id)await selectDataset(did);
  if(revision!==S.scopeRevision||did!==S.dataset?.id)return;S.job=j;renderJob(j);renderJobs();updateScope();
  if(j.kind==='fit'){const maps=await api(`/api/jobs/${id}/maps`);if(revision!==S.scopeRevision||S.job?.id!==id)return;S.maps=maps;renderMapOptions();if(j.coords?.length){const has=j.coords.some(p=>p[0]===S.pixel[0]&&p[1]===S.pixel[1]);if(!has)S.pixel=j.coords[0];}await selectPixel(...S.pixel);}else{S.maps={};S.result=null;renderMapOptions();renderHeatmap();renderSpectrum();renderPixelStats();}
}
async function poll(){
  if(S.polling||S.loading||(S.online===false&&Date.now()<S.nextReconnect))return;S.polling=true;
  try{
    if(!S.initialized){const reconnecting=S.online===false;await loadWorkbench();if(reconnecting&&S.config.max_parallel_workers!==undefined)tell('连接已恢复，数据和计算记录已同步。');return;}
    if(S.online===false){
      await api('/api/health',undefined,3000);
      await refreshJobs();
      const id=S.jobs.find(j=>j.id===S.running)?.id||S.job?.id;if(id)await selectJob(id);
      tell(S.job?.status==='interrupted'?'连接已恢复。上次任务已中断，已保存像素可查看；未完成计算不会自动续跑。':'连接已恢复，计算记录已同步。',S.job?.status==='interrupted');
      S.nextHealth=Date.now()+10000;
    }else if(Date.now()>=S.nextHealth){await api('/api/health',undefined,3000);S.nextHealth=Date.now()+10000;}
    if(!S.running)return;const revision=S.scopeRevision,j=await api(`/api/jobs/${S.running}`);if(j.id!==S.running)return;const selected=jobInScope(j)&&S.job?.id===j.id;const oldCount=S.job?.completed_pixels;
    if(selected){S.job=j;renderJob(j);if(j.kind==='fit'&&(oldCount!==j.completed_pixels||!ACTIVE.includes(j.status))){const maps=await api(`/api/jobs/${j.id}/maps`);if(revision!==S.scopeRevision||S.job?.id!==j.id)return;S.maps=maps;renderMapOptions();await loadPixelResult();renderHeatmap();renderSpectrum();}}
    if(!ACTIVE.includes(j.status)){
      S.running=null;busy(false);await refreshJobs();
      if(jobInScope(j)&&!S.handled.has(j.id)){S.handled.add(j.id);if(j.kind==='preprocess'&&j.output_dataset_id){await selectDataset(j.output_dataset_id);if(jobInScope(j)&&S.dataset?.id===j.output_dataset_id)tell(`预处理完成，用时 ${duration(j.elapsed_seconds)}。已切换至处理后的数据。`);}else if(j.status==='failed')tell(j.error||j.message,true);else if(j.status==='completed_with_errors')tell(j.message,true);else if(j.status==='completed')tell(`计算完成，用时 ${duration(j.elapsed_seconds)}。可点击像素查看诊断并下载结果。`);}
    }
  }catch(e){tell(e.message,true);}finally{S.polling=false;}
}
async function importPath(path){
  busy(true);$('import-status').textContent='正在读取与保存数据…';
  try{const d=await api('/api/import',{path});await selectDataset(d.id,d);await refreshJobs();$('import-dialog').close();$('import-status').textContent='';tell(`已载入 ${d.name} · ${fmt(d.pixels,0)} 像素 × ${d.shape[2]} 通道`);}catch(e){$('import-status').textContent=e.message;throw e;}finally{busy(false);}
}
function attachEvents(){
  $('notice').querySelector('button').onclick=()=>$('notice').classList.add('hidden');
  $('open-import').onclick=()=>{$('import-dialog').showModal();$('file-path').focus();};$('close-import').onclick=()=>$('import-dialog').close();
  $('import-form').onsubmit=e=>{e.preventDefault();action(()=>importPath($('file-path').value));};
  $('load-reference').onclick=()=>action(()=>importPath(S.config.reference_path));
  $('file-upload').onchange=()=>action(async()=>{const file=$('file-upload').files[0];if(!file)return;busy(true);$('import-status').textContent='正在上传与读取…';try{const form=new FormData();form.append('file',file);const body=await requestJSON('/api/upload',{method:'POST',body:form},0);await selectDataset(body.id,body);await refreshJobs();$('import-dialog').close();$('import-status').textContent='';tell(`已载入 ${body.name}`);}finally{busy(false);$('file-upload').value='';}});
  $('dataset-select').onchange=()=>action(()=>selectDataset($('dataset-select').value));$('jump-pixel').onclick=()=>action(()=>selectPixel(Number($('pixel-y').value)-1,Number($('pixel-x').value)-1));
  for(const id of ['pixel-y','pixel-x'])$(id).onkeydown=e=>{if(e.key==='Enter')$('jump-pixel').click();};
  $('map-select').onchange=renderHeatmap;$('scope').onchange=updateScope;for(const id of ['roi-y0','roi-x0','roi-y1','roi-x1'])$(id).oninput=updateScope;
  $('execution-mode').onchange=updateScope;$('fit-stage').onchange=updateScope;
  for(const id of ['parallel-workers','stripe-count'])$(id).oninput=()=>{updateScope();try{localStorage.setItem('zest.parallel.scheduling',JSON.stringify(parallelOptions()));}catch{}};
  $('backend').onchange=()=>{const gpu=$('backend').value==='gpu';$('execution-mode').options[0].disabled=gpu;if(gpu)$('execution-mode').value='serial';$('backend-note').textContent=gpu?'GPU 保留原混合计算路径；拟合使用串行模式，预处理仍可用 GPU。':'CPU 双精度求解；支持竖条并行和串行对照。';updateScope();};
  $('run-fit').onclick=()=>action(startFit);$('preprocess').onclick=()=>action(preprocess);$('cancel-job').onclick=()=>action(async()=>{if(S.job)renderJob(await api(`/api/jobs/${S.job.id}/cancel`,{}));});
  $('refresh-jobs').onclick=()=>action(refreshJobs);
  $('plot-tabs').onclick=e=>{const b=e.target.closest('[data-view]');if(!b)return;S.view=b.dataset.view;for(const item of $('plot-tabs').children)item.classList.toggle('active',item===b);renderSpectrum();};
  $('scale-switch').onclick=e=>{const b=e.target.closest('[data-scale]');if(!b)return;S.scale=b.dataset.scale;for(const item of $('scale-switch').children)item.classList.toggle('active',item===b);renderSpectrum();};
  $('download-plot').onclick=()=>action(()=>Plotly.downloadImage('spectrum',{format:'png',filename:`EELS-ZEST-Y${S.pixel[0]+1}-X${S.pixel[1]+1}-${S.view}`,width:1400,height:650,scale:2}));
  for(const b of document.querySelectorAll('[data-scroll]'))b.onclick=()=>step(b.dataset.scroll);
  $('import-dialog').onclick=e=>{if(e.target===$('import-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}};
}
async function loadWorkbench(){
    S.config=await api('/api/config');
    let scheduling={parallel_workers:S.config.parallel_workers||12,stripe_count:S.config.stripe_count||0};
    try{const saved=JSON.parse(localStorage.getItem('zest.parallel.scheduling'));if(saved&&Number.isInteger(saved.parallel_workers)&&saved.parallel_workers>=1&&saved.parallel_workers<=(S.config.max_parallel_workers||61)&&Number.isInteger(saved.stripe_count)&&saved.stripe_count>=0&&saved.stripe_count<=10000)scheduling=saved;}catch{}
    $('parallel-workers').max=S.config.max_parallel_workers||61;$('parallel-workers').value=scheduling.parallel_workers;$('stripe-count').value=scheduling.stripe_count;
    $('parallel-hint').textContent=`本机 ${S.config.logical_cpus||'—'} 个逻辑线程。条带数 0 = 自动，最多为进程数的 3 倍。`;
    updateScope();
    const h=S.config.hardware;$('device-name').textContent=h.gpu?h.gpu_name.replace('NVIDIA GeForce ',''):'CPU · NumPy / SciPy';$('device-detail').textContent=h.gpu?`${fmt(h.gpu_memory_bytes/1024**3,1)} GB VRAM · CUDA 混合执行`:'未检测到 CUDA · CPU 可用';$('backend').options[1].disabled=!h.gpu;
    if(!h.gpu)$('device-detail').title=h.gpu_error||'';
    $('load-reference').classList.toggle('hidden',!S.config.reference_available);$('file-path').value=S.config.reference_available?S.config.reference_path:'';
    const d=S.config.defaults;$('crop-min').value=d.crop_range[0];$('crop-max').value=d.crop_range[1];$('model-type').value=d.zlp_model_type==='pvoigt'?'pvoigt':'pearson';$('temp-min').value=d.stage2_T_grid[0];$('temp-max').value=d.stage2_T_grid.at(-1);$('temp-step').value=d.stage2_T_grid[1]-d.stage2_T_grid[0];
    const entries=await refreshDatasets();const prior=localStorage.getItem('zest.parallel.dataset');if(entries.length)await selectDataset(entries.some(d=>d.id===prior)?prior:entries[0].id);
    await refreshJobs();if(S.jobs.some(j=>j.id===S.running))await selectJob(S.running);else if(S.dataset){const latest=S.jobs.find(j=>j.kind==='fit'&&j.dataset_id===S.dataset.id);if(latest)await selectJob(latest.id);}
    if(S.config.max_parallel_workers===undefined)tell('代码已更新，后端服务仍是旧版本。请运行 stop_web.ps1，再运行 start_web.ps1 并刷新页面；已有记录可继续查看。',true);
    S.initialized=true;busy(false);S.nextHealth=Date.now()+10000;
}
function init(){
  attachEvents();renderSpectrum();busy(false);
  // Keep polling even when initial startup fails, including an idle page.
  setInterval(poll,1600);poll();
}
document.addEventListener('DOMContentLoaded',init);
