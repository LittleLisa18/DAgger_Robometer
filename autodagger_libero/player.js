const $ = id => document.getElementById(id);
let meta = null, frame = null, generation = 0, playing = false, timer = null;
let wanted = 0, loading = false, threshold = 0.5;
const reasons = {progress_stalled:'进度停滞', progress_regression:'进度持续退步', forced_test:'强制接管（测试）'};
async function get(url) {
  const response = await fetch(url, {cache:'no-store', signal:AbortSignal.timeout(15000)});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || `HTTP ${response.status}`);
  return data;
}
function pause() { playing = false; clearTimeout(timer); $('play').textContent = '播放'; }
function controls() {
  const empty = !meta || !meta.steps;
  for (const id of ['play','prev','next','seek']) $(id).disabled = empty;
  $('takeover').disabled = empty || meta.takeover_step == null || meta.takeover_step >= meta.steps;
}
function draw() {
  const c = $('chart'), w = c.clientWidth, h = 210, d = devicePixelRatio || 1;
  c.width=w*d; c.height=h*d;
  const ctx=c.getContext('2d'); ctx.scale(d,d);
  const samples=meta?.samples || [], max=Math.max(1,meta?.steps || 0);
  const lo=Math.min(0,...samples.map(s=>s.progress)), hi=Math.max(1,...samples.map(s=>s.progress));
  const x=s=>40+s/max*(w-60), y=v=>175-(v-lo)/(hi-lo)*145;
  ctx.font='12px system-ui'; ctx.fillStyle='#9aaebe'; ctx.strokeStyle='#344555';
  for(let i=0;i<=4;i++){let v=lo+(hi-lo)*i/4;ctx.beginPath();ctx.moveTo(40,y(v));ctx.lineTo(w-20,y(v));ctx.stroke();ctx.fillText(v.toFixed(2),0,y(v)+4);}
  ctx.fillText('0',40,200);ctx.fillText(`${max} steps`,w-95,200);
  ctx.setLineDash([4,4]);ctx.strokeStyle='#c5b776';ctx.beginPath();ctx.moveTo(40,y(threshold));ctx.lineTo(w-20,y(threshold));ctx.stroke();ctx.setLineDash([]);
  for(const [key,color] of [['progress','#70d1ca'],['success_probability','#ad9eff']]){
    ctx.strokeStyle=color;ctx.beginPath();samples.forEach((s,i)=>i?ctx.lineTo(x(s.step),y(s[key])):ctx.moveTo(x(s.step),y(s[key])));ctx.stroke();
  }
  for(const [step,color] of [[meta?.takeover_step,'#ffaa55'],[frame?.step,'#ffffff']]){
    if(step!=null){ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(x(step),20);ctx.lineTo(x(step),175);ctx.stroke();}
  }
}
function render(data) {
  frame=data;
  $('image').src=data.image; $('image2').src=data.image2;
  $('actor').textContent=data.collect; $('actor').className=data.collect==='teacher'?'teacher':'';
  $('position').textContent=`step ${data.step} · 帧 ${data.index+1}/${data.count} · 动作前观测`;
  $('seek').value=data.index;
  $('action').textContent=JSON.stringify(data.action.map(x=>+x.toFixed(5)));
  $('state').textContent=JSON.stringify(data.state.map(x=>+x.toFixed(5)));
  const scores=(meta.samples || []).filter(s=>s.step<=data.step), latest=scores.at(-1);
  $('score').textContent='青：progress · 紫：成功概率 · 虚线：成功阈值 · 橙：接管 · 白：当前帧。'+(latest?`截至 step ${latest.step} 的评分：progress ${latest.progress.toFixed(3)}，成功概率 ${latest.success_probability.toFixed(3)}`:'当前帧之前尚无评分');
  draw();
}
async function pump() {
  if(loading || !meta?.steps)return;
  loading=true;
  const token=generation, id=meta.episode_id, index=wanted;
  try {
    const data=await get(`/api/replay/${encodeURIComponent(id)}/${index}`);
    // Discard responses from a previously selected episode or superseded seek.
    if(token!==generation || index!==wanted)return;
    await Promise.all(['image','image2'].map(key=>new Promise((resolve,reject)=>{const img=new Image();img.onload=resolve;img.onerror=reject;img.src=data[key];})));
    if(token!==generation || index!==wanted)return;
    render(data);$('error').textContent='';
    if(playing){if(index+1>=data.count)pause();else timer=setTimeout(()=>seek(index+1),1000/Number($('speed').value));}
  } catch(error) { if(token===generation){pause();$('error').textContent=`加载失败：${error.message || error}。可拖动或点击播放重试。`;} }
  finally {loading=false;if(meta?.steps && (token!==generation || index!==wanted))pump();}
}
function seek(index) {if(!meta?.steps)return;wanted=Math.max(0,Math.min(meta.steps-1,index));clearTimeout(timer);pump();}
async function selectEpisode(id) {
  pause();const token=++generation;meta=null;frame=null;controls();
  for(const key of ['image','image2'])$(key).removeAttribute('src');
  for(const key of ['action','state','score','reason','notice','metadata','task'])$(key).textContent='';
  $('actor').textContent='—';$('actor').className='';$('position').textContent='正在加载…';draw();
  try {
    const result=await get('/api/episodes/'+encodeURIComponent(id));if(token!==generation)return;
    meta=result;$('task').textContent=meta.task;$('metadata').textContent=JSON.stringify(meta,null,2);
    $('notice').textContent=(meta.test_only?'测试数据 · ':'')+(meta.score_source==='simulated'?'模拟评分器 · ':'')+(meta.accepted_for_distillation?'筛选通过':'未通过蒸馏筛选');
    $('reason').textContent=meta.takeover_step==null?'未接管':`${reasons[meta.takeover_reason] || meta.takeover_reason || '原因未记录'} · step ${meta.takeover_step}`;
    $('seek').max=Math.max(0,meta.steps-1);$('seek').value=0;controls();$('error').textContent='';
    if(meta.steps)seek(0);else {$('position').textContent='该 episode 没有已保存动作帧';draw();}
  } catch(error){if(token===generation)$('error').textContent=error.message;}
}
$('episode').onchange=()=>selectEpisode($('episode').value);
$('play').onclick=()=>{if(playing)pause();else{playing=true;$('play').textContent='暂停';seek(frame?.index===meta.steps-1?0:wanted);}};
$('prev').onclick=()=>{pause();seek(wanted-1);};$('next').onclick=()=>{pause();seek(wanted+1);};
$('seek').oninput=()=>{pause();seek(Number($('seek').value));};
$('takeover').onclick=()=>{pause();seek(meta.takeover_step);};
window.addEventListener('resize',draw);
async function init(){
  controls();
  try{
    const [episodes,state]=await Promise.all([get('/api/replay-episodes'),get('/api/state')]);threshold=state.success_threshold;
    $('episode').replaceChildren();
    for(const episode of episodes){const option=document.createElement('option');option.value=episode.episode_id;option.textContent=episode.episode_id;$('episode').append(option);}
    const requested=new URLSearchParams(location.search).get('episode');
    if(requested && episodes.some(e=>e.episode_id===requested))$('episode').value=requested;
    if(episodes.length)await selectEpisode($('episode').value);else $('position').textContent='暂无已保存 episode';
  }catch(error){$('error').textContent=error.message;}
}
init();
