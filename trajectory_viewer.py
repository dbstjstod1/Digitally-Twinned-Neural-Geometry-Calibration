"""Write an offline, equal-scale interactive viewer of physical source positions."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path


def write_trajectory_html(path, *, theta_deg, truth_xyz, estimated_xyz, aligned_xyz,
                          title, registration_note):
    """Write a self-contained HTML viewer; xyz arrays use one common frame in mm.

    No fitting, alignment, smoothing, resampling, or residual magnification occurs
    here. All supplied views are drawn, with an orthographic camera and the same
    scale for x, y, and z. Array-like inputs need no particular Python package.
    """
    theta = [float(value) for value in theta_deg]
    if len(theta) < 2 or not all(math.isfinite(value) for value in theta):
        raise ValueError("theta_deg must contain at least two finite angles.")
    curves = []
    for name, values in (("GT", truth_xyz), ("Raw estimate", estimated_xyz),
                         ("Phantom-aligned estimate", aligned_xyz)):
        xyz = [[float(value) for value in row] for row in values]
        if len(xyz) != len(theta) or any(len(row) != 3 for row in xyz):
            raise ValueError(f"{name} must have shape (len(theta_deg), 3).")
        if not all(math.isfinite(value) for row in xyz for value in row):
            raise ValueError(f"{name} must contain only finite coordinates.")
        curves.append({"name": name, "xyz": xyz})
    data = json.dumps({"theta": theta, "curves": curves}, allow_nan=False,
                      separators=(",", ":")).replace("<", "\\u003c")
    page = _PAGE.replace("__TITLE__", html.escape(str(title)))
    page = page.replace("__NOTE__", html.escape(str(registration_note)))
    page = page.replace("__DATA__", data)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    return destination


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#202c3b;font:15px system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:20px}h1{font-size:24px;margin:0 0 10px}p{line-height:1.5;margin:9px 0}
.panel{background:white;border:1px solid #dbe1e8;border-radius:9px;padding:15px;margin-top:15px}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:16px}.controls label{cursor:pointer}
button{border:1px solid #bdc6d2;border-radius:5px;background:#fff;padding:7px 12px;cursor:pointer}
button:hover{background:#eef2f7}.swatch{display:inline-block;width:22px;height:3px;vertical-align:middle;margin:0 5px}
canvas{display:block;width:100%;height:min(65vh,650px);min-height:320px;touch-action:none;cursor:grab}
canvas:active{cursor:grabbing}canvas:focus-visible{outline:2px solid #245c9d}
.muted{color:#526171;font-size:13px}.slider{display:flex;align-items:center;gap:14px;margin:12px 0}
input[type=range]{flex:1;min-width:80px}output{font-variant-numeric:tabular-nums;min-width:230px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;text-align:right;border-bottom:1px solid #e8ecf0}
th:first-child,td:first-child{text-align:left}thead{font-size:13px;color:#526171}.table-wrap{overflow:auto}
@media(max-width:620px){main{padding:10px}h1{font-size:20px}.slider{flex-wrap:wrap}output{min-width:100%}}
</style></head><body><main><h1>__TITLE__</h1>
<p>Physical source coordinates, millimetres. Drag to rotate; scroll or pinch to zoom. Arrow keys rotate; +/− zoom.</p>
<p class="muted">Orthographic projection, equal x/y/z scale. Every supplied view is drawn. Overlapping curves are expected; errors below use the original 3D coordinates.</p>
<div class="panel"><div class="controls" id="legend"></div>
<canvas id="plot" tabindex="0" aria-label="Interactive 3D source trajectories, equal physical axis scale"></canvas>
<div class="slider"><label for="view">Selected view</label><input id="view" type="range" min="0" step="1"><output id="view-label"></output></div>
<div class="table-wrap"><table><thead><tr><th>Trajectory</th><th>x (mm)</th><th>y (mm)</th><th>z (mm)</th><th>Distance to GT (mm)</th></tr></thead><tbody id="coordinates"></tbody></table></div></div>
<div class="panel"><strong>Registration</strong><p>__NOTE__</p>
<p class="muted">Displayed alignment is supplied by the analysis. This viewer performs no additional fitting and never scales or exaggerates trajectory differences.</p></div>
<noscript><p>Enable JavaScript to view this self-contained interactive plot.</p></noscript>
</main><script id="trajectory-data" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('trajectory-data').textContent);
const canvas=document.getElementById('plot'), ctx=canvas.getContext('2d');
const slider=document.getElementById('view'), readout=document.getElementById('view-label');
const colors=['#202b37','#df7420','#2479bf'], visible=[true,true,true];
const defaults={yaw:-0.55,pitch:0.55,zoom:1}; let camera={...defaults}, selected=0, width=0,height=0;
let maximum=0; for(const curve of data.curves) for(const xyz of curve.xyz) for(const v of xyz) maximum=Math.max(maximum,Math.abs(v));
const magnitude=10**Math.floor(Math.log10(Math.max(maximum,1))), tick=magnitude*(maximum/magnitude>5?2:1);
const extent=Math.max(tick,Math.ceil(maximum/tick)*tick), corners=[];
for(const x of [-extent,extent])for(const y of [-extent,extent])for(const z of [-extent,extent])corners.push([x,y,z]);
const legend=document.getElementById('legend');
data.curves.forEach((curve,index)=>{
  const label=document.createElement('label'), check=document.createElement('input'), swatch=document.createElement('span');
  check.type='checkbox';check.checked=true;check.addEventListener('change',()=>{visible[index]=check.checked;draw();});
  swatch.className='swatch';swatch.style.background=colors[index];
  label.append(check,swatch,document.createTextNode(curve.name));legend.append(label);
});
const reset=document.createElement('button');reset.textContent='Reset view';reset.addEventListener('click',()=>{camera={...defaults};draw();});legend.append(reset);
const status=document.createElement('span');status.className='muted';legend.append(status);
slider.max=data.theta.length-1;slider.value=0;
function rotate(p){
  const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch);
  const x=cy*p[0]-sy*p[1], y=sy*p[0]+cy*p[1];return [x,cp*p[2]-sp*y,cp*y+sp*p[2]];
}
function project(p){const r=rotate(p),scale=Math.min(width,height)*0.245/extent*camera.zoom;return [width/2+r[0]*scale,height/2-r[1]*scale,r[2]];}
function line(a,b,color,widthPx=1,dash=[]){
  const u=project(a),v=project(b);ctx.strokeStyle=color;ctx.lineWidth=widthPx;ctx.setLineDash(dash);
  ctx.beginPath();ctx.moveTo(u[0],u[1]);ctx.lineTo(v[0],v[1]);ctx.stroke();ctx.setLineDash([]);
}
function draw(){
  ctx.clearRect(0,0,width,height);ctx.fillStyle='#fff';ctx.fillRect(0,0,width,height);
  for(let i=0;i<8;i++)for(let j=i+1;j<8;j++)if(corners[i].filter((v,k)=>v!==corners[j][k]).length===1)line(corners[i],corners[j],'#e1e6ed');
  const axisColors=['#b44747','#368058','#4d64ae'];
  for(let axis=0;axis<3;axis++){
    const a=[0,0,0],b=[0,0,0];a[axis]=-extent;b[axis]=extent;line(a,b,axisColors[axis],1.1);
    const end=project(b);ctx.fillStyle=axisColors[axis];ctx.font='bold 14px system-ui';ctx.fillText('xyz'[axis]+' (mm)',end[0]+6,end[1]-6);
    for(let value=-extent;value<=extent+tick/2;value+=tick){
      if(value===0)continue;const point=[0,0,0];point[axis]=value;const p=project(point);
      ctx.fillRect(p[0]-1.5,p[1]-1.5,3,3);ctx.font='11px system-ui';ctx.fillText(value.toLocaleString('en-US'),p[0]+5,p[1]+13);
    }
  }
  const origin=project([0,0,0]);ctx.fillStyle='#687484';ctx.fillText('0',origin[0]+5,origin[1]+12);
  data.curves.forEach((curve,index)=>{
    if(!visible[index])return;ctx.strokeStyle=colors[index];ctx.lineWidth=index===0?2.5:1.7;ctx.globalAlpha=index===0?0.65:0.9;
    ctx.setLineDash(index===1?[6,3]:[]);ctx.beginPath();
    curve.xyz.forEach((xyz,i)=>{const p=project(xyz);if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();ctx.setLineDash([]);
  });ctx.globalAlpha=1;
  // Concentric marker radii reveal coincident points without changing coordinates.
  data.curves.forEach((curve,index)=>{if(!visible[index])return;const p=project(curve.xyz[selected]);
    ctx.strokeStyle=colors[index];ctx.lineWidth=2;ctx.beginPath();ctx.arc(p[0],p[1],9-2*index,0,2*Math.PI);ctx.stroke();});
  ctx.fillStyle='#526171';ctx.font='12px system-ui';ctx.fillText('Selected view: concentric rings mark actual positions',12,height-12);
  status.textContent='All '+data.theta.length+' views · zoom '+camera.zoom.toFixed(2)+'×';
}
function updateSelection(){
  selected=Number(slider.value);readout.textContent='View '+selected+' / '+(data.theta.length-1)+' · θ = '+data.theta[selected].toFixed(3)+'°';
  const body=document.getElementById('coordinates');body.replaceChildren();const truth=data.curves[0].xyz[selected];
  data.curves.forEach((curve,index)=>{const row=document.createElement('tr'),p=curve.xyz[selected];
    const error=Math.hypot(...p.map((value,i)=>value-truth[i]));
    [curve.name,...p.map(v=>v.toFixed(4)),error.toFixed(6)].forEach((text,i)=>{const cell=document.createElement('td');cell.textContent=text;if(i===0)cell.style.color=colors[index];row.append(cell);});body.append(row);
  });draw();
}
function resize(){const box=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;width=box.width;height=box.height;
  canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);draw();}
slider.addEventListener('input',updateSelection);
const pointers=new Map();let previousDistance=null;
canvas.addEventListener('pointerdown',event=>{canvas.setPointerCapture(event.pointerId);pointers.set(event.pointerId,[event.clientX,event.clientY]);});
canvas.addEventListener('pointermove',event=>{
  if(!pointers.has(event.pointerId))return;const previous=pointers.get(event.pointerId);pointers.set(event.pointerId,[event.clientX,event.clientY]);
  if(pointers.size===1){camera.yaw+=(event.clientX-previous[0])*0.008;camera.pitch+=(event.clientY-previous[1])*0.008;}
  else{const p=[...pointers.values()],distance=Math.hypot(p[0][0]-p[1][0],p[0][1]-p[1][1]);if(previousDistance&&distance)camera.zoom*=distance/previousDistance;previousDistance=distance;}
  camera.zoom=Math.max(0.2,Math.min(15,camera.zoom));draw();
});
function release(event){pointers.delete(event.pointerId);previousDistance=null;}
canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);canvas.addEventListener('lostpointercapture',release);
canvas.addEventListener('wheel',event=>{event.preventDefault();camera.zoom=Math.max(0.2,Math.min(15,camera.zoom*Math.exp(-event.deltaY*0.001)));draw();},{passive:false});
canvas.addEventListener('keydown',event=>{const key=event.key;if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','=','Home'].includes(key))return;event.preventDefault();
  if(key==='Home')camera={...defaults};if(key==='ArrowLeft')camera.yaw-=0.1;if(key==='ArrowRight')camera.yaw+=0.1;if(key==='ArrowUp')camera.pitch-=0.1;if(key==='ArrowDown')camera.pitch+=0.1;
  if(key==='+'||key==='=')camera.zoom*=1.1;if(key==='-')camera.zoom/=1.1;camera.zoom=Math.max(0.2,Math.min(15,camera.zoom));draw();});
window.addEventListener('resize',resize);resize();updateSelection();
</script></body></html>'''
