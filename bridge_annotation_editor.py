from pathlib import Path
from google.colab import drive
from google.colab.output import eval_js
from PIL import Image, ImageDraw, ImageFont
import base64, csv, json, random, re

# Uses DATASET, SPLIT_PATHS, LABELS from the earlier setup cells.
name = re.sub(r'[^A-Za-z0-9_-]+', '_', STUDENT.strip()).strip('_')
if not name:
    raise ValueError('Please enter your name.')

drive.mount('/content/drive')
imgdir = DATASET / SPLIT_PATHS['train'] / 'images'
labeldir = DATASET / SPLIT_PATHS['train'] / 'labels'
all_imgs = sorted(p for p in imgdir.iterdir()
                  if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
OUT = Path('/content/drive/MyDrive/BridgeVision_AI/Reviews') / name / 'practice' / 'train'
LOG = OUT / 'review_log.csv'
for folder in ('annotated_previews', 'annotations_json', 'labels_detect', 'labels_segment'):
    (OUT / folder).mkdir(parents=True, exist_ok=True)

# Assign five unique images. Existing 3-image assignments are extended, not replaced.
N_IMAGES = 5
assignment = OUT / 'assigned_images.json'
previous = json.loads(assignment.read_text()) if assignment.exists() else []
if not isinstance(previous, list) or len(previous) != len(set(previous)) or len(previous) > N_IMAGES:
    raise ValueError('Saved assigned_images.json is invalid. Check its image list.')
if any(not (imgdir / filename).is_file() for filename in previous):
    raise FileNotFoundError('A previously assigned image is missing from the dataset.')

# Avoid images already displayed in STEP 2, when possible.
shown = {pair[1].name for pair in globals().get('chosen', [])
         if isinstance(pair, (tuple, list)) and len(pair) == 2
         and hasattr(pair[1], 'name')}
candidates = [p for p in all_imgs if p.name not in shown and p.name not in previous]
rng = random.Random(0)  # Fixed seed: same first assignment if dataset and exclusions match.
rng.shuffle(candidates)

# Prefer visually different images (not necessarily higher-quality ones).
def fingerprint(path):
    with Image.open(path) as im:
        pixels = list(im.convert('L').resize((9, 8)).getdata())
    return sum((pixels[i] > pixels[i - 1]) << (i - i // 9 - 1)
               for i in range(1, 72) if i % 9)

selected = [(imgdir / filename, fingerprint(imgdir / filename)) for filename in previous]
if len(selected) < N_IMAGES:
    pool = []
    for p in candidates[:80]:
        try:
            pool.append((p, fingerprint(p)))
        except OSError:
            pass
    if len(pool) < N_IMAGES - len(selected):
        raise ValueError('Not enough readable, unused training images for five assignments.')
    if not selected:
        selected.append(pool.pop(0))
    while len(selected) < N_IMAGES:
        best = max(pool, key=lambda item:
                   min((item[1] ^ old[1]).bit_count() for old in selected))
        selected.append(best)
        pool.remove(best)
    assignment.write_text(json.dumps([p.name for p, _ in selected], indent=2))

assigned = [p for p, _ in selected]
print(f'🌉 {name}: {len(assigned)} distinct images assigned. Draw and click “Save & next”.')

EDITOR = r'''
(async () => new Promise(resolve => {
  const d = PAYLOAD;
  const panel = document.createElement('div');
  panel.style = 'font:14px Arial,sans-serif;max-width:920px;color:#222;padding:12px;background:#fafafa;border:1px solid #ddd;border-radius:12px';
  panel.innerHTML = `
    <style>
      .bv-toolbar { display:flex;flex-wrap:wrap;align-items:center;gap:8px; }
      .bv-tool { background:#fff;border:1px solid #cbd5e1;border-radius:8px;
        padding:9px 13px;font:600 14px Arial,sans-serif;color:#334155;cursor:pointer; }
      .bv-tool:hover { background:#f1f5f9; }
      .bv-tool.active { background:#2563eb;border-color:#2563eb;color:#fff; }
      .bv-action { background:#fff;border:1px solid #cbd5e1;border-radius:7px;
        padding:8px 11px;color:#334155;cursor:pointer; }
      .bv-action:hover { background:#f1f5f9; }
      .bv-action:disabled { opacity:.45;cursor:not-allowed; }
      .bv-save { background:#16834b;border-color:#16834b;color:#fff;font-weight:600; }
      .bv-save:hover { background:#11683b; }
    </style>
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:11px">
      <b style="font-size:16px">🌉 ${d.title}</b>
      <label style="font-weight:600">Bridge component:
        <select id="cls" style="margin-left:5px;padding:7px;border:1px solid #cbd5e1;border-radius:7px;font-size:14px"></select>
      </label>
    </div>
    <div class="bv-toolbar" style="margin-bottom:10px" role="group" aria-label="Annotation drawing tools">
      <button type="button" class="bv-tool" data-tool="select">↖ Select / Edit</button>
      <button type="button" class="bv-tool active" data-tool="polygon" aria-pressed="true">⬠ Polygon</button>
      <button type="button" class="bv-tool" data-tool="box">▭ Box</button>
      <span style="width:1px;height:25px;background:#ddd;margin:0 3px"></span>
      <button type="button" class="bv-action" id="finish">Finish polygon</button>
      <button type="button" class="bv-action" id="undo">↶ Undo</button>
      <button type="button" class="bv-action" id="del">Delete selected</button>
    </div>
    <div id="hint" style="color:#475569;margin-bottom:9px;line-height:1.5">
      Click along the edge of a bridge component, then click the <b style="color:#009c50">green starting point</b> to finish.
    </div>
    <canvas id="canvas" style="width:100%;height:auto;display:block;touch-action:none;border:1px solid #bbb;border-radius:6px;cursor:crosshair"></canvas>
    <div style="display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin-top:11px">
      <button type="button" class="bv-action bv-save" id="save">✓ Save & next</button>
      <button type="button" class="bv-action" id="skip">Skip</button>
      <button type="button" class="bv-action" id="stop">Stop</button>
      <span id="count" style="margin-left:auto;color:#475569"></span>
    </div>`;
  document.body.appendChild(panel);
  const $ = id => panel.querySelector('#' + id);
  const canvas = $('canvas'), ctx = canvas.getContext('2d');
  const img = new Image();
  const colors = ['#2579c7','#e88b27','#279a57','#9454ba','#e05757'];
  let shapes = [], history = [], vertices = [], moving = null, boxStart = null, selected = -1;
  const copy = x => JSON.parse(JSON.stringify(x));
  d.names.forEach((n,i) => $('cls').add(new Option(n,i)));
  let currentTool = 'polygon';
  const tool = () => currentTool;
  const updateToolbar = () => {
    panel.querySelectorAll('[data-tool]').forEach(btn => {
      const active = btn.dataset.tool === currentTool;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-pressed', String(active));
    });
    $('finish').style.display = currentTool === 'polygon' ? '' : 'none';
    $('hint').innerHTML = {
      polygon: 'Click along the edge of a bridge component, then click the <b style="color:#009c50">green starting point</b> to finish.',
      box: 'Click and drag to draw a rectangle around a bridge component.',
      select: 'Click an existing shape to select it. You can change its component or delete it.'
    }[currentTool];
    canvas.style.cursor = currentTool === 'select' ? 'default' : 'crosshair';
  };
  const pos = e => { const r=canvas.getBoundingClientRect(); return [
    Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),
    Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]; };
  const scale = () => canvas.width / (canvas.getBoundingClientRect().width || canvas.width);
  const closeEnough = p => vertices.length>=3 && Math.hypot(
    (p[0]-vertices[0][0])*canvas.getBoundingClientRect().width,
    (p[1]-vertices[0][1])*canvas.getBoundingClientRect().height) < 20;

  function inside(p,s) {
    if(s.kind==='box') {const [x,y,w,h]=s.data;
      return Math.abs(p[0]-x)<=w/2 && Math.abs(p[1]-y)<=h/2;}
    let hit=false, a=s.data;
    for(let i=0,j=a.length-2;i<a.length;j=i,i+=2) {
      const xi=a[i], yi=a[i+1], xj=a[j], yj=a[j+1];
      if((yi>p[1])!==(yj>p[1]) && p[0]<(xj-xi)*(p[1]-yi)/(yj-yi)+xi) hit=!hit;
    }
    return hit;
  }
  function paint() {
    if(!img.complete || !img.naturalWidth) return;
    const w=canvas.width, h=canvas.height, z=scale();
    ctx.clearRect(0,0,w,h); ctx.drawImage(img,0,0,w,h);
    shapes.forEach((s,i) => {
      const color=colors[s.cls%colors.length];
      ctx.strokeStyle=color;ctx.lineWidth=(i===selected?3:2)*z;
      let x,y;
      if(s.kind==='box') {const [xc,yc,bw,bh]=s.data;
        x=(xc-bw/2)*w;y=(yc-bh/2)*h;ctx.strokeRect(x,y,bw*w,bh*h);
      } else {const a=s.data;
        x=Math.min(...a.filter((_,j)=>j%2===0))*w;
        y=Math.min(...a.filter((_,j)=>j%2===1))*h;
        ctx.beginPath();ctx.moveTo(a[0]*w,a[1]*h);
        for(let j=2;j<a.length;j+=2)ctx.lineTo(a[j]*w,a[j+1]*h);
        ctx.closePath();ctx.stroke();
      }
      const label=d.names[s.cls];ctx.font=`bold ${14*z}px Arial`;
      const tw=ctx.measureText(label).width+8*z, ty=Math.max(0,y-19*z);
      ctx.fillStyle=color;ctx.fillRect(x,ty,tw,19*z);
      ctx.fillStyle='white';ctx.fillText(label,x+4*z,ty+14*z);
    });
    if(boxStart&&moving) {
      ctx.strokeStyle='#ff3155';ctx.lineWidth=2*z;
      ctx.strokeRect(Math.min(boxStart[0],moving[0])*w,Math.min(boxStart[1],moving[1])*h,
        Math.abs(boxStart[0]-moving[0])*w,Math.abs(boxStart[1]-moving[1])*h);
    }
    if(vertices.length) {
      const pts=[...vertices];if(moving&&tool()==='polygon')pts.push(closeEnough(moving)?vertices[0]:moving);
      ctx.strokeStyle='#ff3155';ctx.lineWidth=2*z;ctx.beginPath();
      pts.forEach((p,i)=>i?ctx.lineTo(p[0]*w,p[1]*h):ctx.moveTo(p[0]*w,p[1]*h));ctx.stroke();
      vertices.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0]*w,p[1]*h,6*z,0,Math.PI*2);
        ctx.fillStyle=i===0?'#00bb59':'#ff3155';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1.5*z;ctx.stroke();});
    }
    $('count').textContent=`${shapes.length} shapes`;
  }
  function finish() {
    if(vertices.length<3)return;
    history.push(copy(shapes));shapes.push({kind:'polygon',cls:+$('cls').value,data:vertices.flat()});
    selected=shapes.length-1;vertices=[];moving=null;paint();
  }
  canvas.onpointerdown=e=>{
    e.preventDefault();const p=pos(e);
    if(tool()==='polygon'){if(closeEnough(p))finish();else {vertices.push(p);moving=p;paint();}}
    else if(tool()==='box'){boxStart=p;moving=p;canvas.setPointerCapture(e.pointerId);paint();}
    else {selected=-1;for(let i=shapes.length-1;i>=0;i--)
      if(inside(p,shapes[i])){selected=i;$('cls').value=shapes[i].cls;break;}paint();}
  };
  canvas.onpointermove=e=>{moving=pos(e);paint();};
  canvas.onpointerleave=()=>{if(!boxStart){moving=null;paint();}};
  canvas.onpointerup=e=>{
    if(!boxStart)return;const p=pos(e),a=boxStart;
    const w=Math.abs(p[0]-a[0]),h=Math.abs(p[1]-a[1]);
    if(w>.008&&h>.008){history.push(copy(shapes));shapes.push({kind:'box',cls:+$('cls').value,
      data:[(p[0]+a[0])/2,(p[1]+a[1])/2,w,h]});selected=shapes.length-1;}
    boxStart=null;moving=null;paint();
  };
  $('finish').onclick=finish;
  $('undo').onclick=()=>{if(vertices.length)vertices.pop();else if(history.length){shapes=history.pop();selected=-1;}paint();};
  $('del').onclick=()=>{if(selected>=0){history.push(copy(shapes));shapes.splice(selected,1);selected=-1;paint();}};
  $('cls').onchange=()=>{if(tool()==='select'&&selected>=0){history.push(copy(shapes));shapes[selected].cls=+$('cls').value;paint();}};
  panel.querySelectorAll('[data-tool]').forEach(btn => btn.onclick = () => {
    if (currentTool === btn.dataset.tool) return;
    currentTool = btn.dataset.tool;
    vertices = []; boxStart = null; moving = null;
    updateToolbar(); paint();
  });
  updateToolbar();
  function done(status){panel.remove();resolve({status,shapes});}
  $('save').onclick=()=>{
    if(vertices.length)return alert('Close or finish the polygon first.');
    if(!shapes.length)return alert('Please annotate at least one bridge component.');
    done('edited');};
  $('skip').onclick=()=>done('skipped');
  $('stop').onclick=()=>done('stopped');
  img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;paint();
    if(typeof google!=='undefined' && google?.colab?.output?.setIframeHeight)
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight,true);};
  img.onerror=()=>{$('count').textContent='Image failed to load';};
  img.src=d.image;
}))()
'''

def annotated_preview(path, shapes, dest):
    with Image.open(path) as src:
        pic = src.convert('RGB')
    w,h = pic.size
    draw = ImageDraw.Draw(pic)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', max(15, min(w,h)//45))
    except OSError:
        font = ImageFont.load_default()
    colors = ['#2579c7','#e88b27','#279a57','#9454ba','#e05757']
    # Keep the student's name visible in every exported image.
    draw.text((10,10), name, font=font, fill='white', stroke_width=2, stroke_fill='black')
    for s in shapes:
        a=s['data'];color=colors[s['cls']%len(colors)]
        if s['kind']=='box':
            x,y,bw,bh=a; x1=(x-bw/2)*w;y1=(y-bh/2)*h
            x2=(x+bw/2)*w;y2=(y+bh/2)*h
            draw.rectangle((x1,y1,x2,y2),outline=color,width=max(3,min(w,h)//250))
        else:
            points=[(a[i]*w,a[i+1]*h) for i in range(0,len(a),2)]
            draw.line(points+[points[0]],fill=color,width=max(3,min(w,h)//250))
            x1=min(p[0] for p in points);y1=min(p[1] for p in points)
        draw.text((int(x1),max(0,int(y1)-22)),str(LABELS[s['cls']]),font=font,
                  fill=color,stroke_width=1,stroke_fill='white')
    pic.save(dest,quality=90)

for i,p in enumerate(assigned,1):
    target = OUT / 'annotations_json' / f'{p.stem}.json'
    if target.exists():
        print(f'✅ {i}/{N_IMAGES} already saved: {p.name}')
        continue
    mime = 'image/png' if p.suffix.lower()=='.png' else 'image/jpeg'
    if p.suffix.lower()=='.webp': mime='image/webp'
    encoded = base64.b64encode(p.read_bytes()).decode()
    payload = {'title':f'{name} — image {i}/{N_IMAGES}', 'names':list(LABELS),
               'image':f'data:{mime};base64,{encoded}'}
    response = eval_js(EDITOR.replace('PAYLOAD',json.dumps(payload)))
    status = response['status']
    if status=='stopped':
        print('Stopped; saved work is still in your Google Drive.')
        break
    shapes = [s for s in response['shapes']
              if s['kind'] in ('box','polygon') and 0<=int(s['cls'])<len(LABELS)]
    if status=='edited':
        target.write_text(json.dumps({'image':p.name,'shapes':shapes},indent=2))
        annotated_preview(p,shapes,OUT/'annotated_previews'/f'{p.stem}.jpg')
        # YOLO detection labels (polygons become boxes).
        detection=[]; segmentation=[]
        for s in shapes:
            a=s['data'];cls=int(s['cls'])
            if s['kind']=='polygon':
                xs=a[::2];ys=a[1::2];box=[(min(xs)+max(xs))/2,
                    (min(ys)+max(ys))/2,max(xs)-min(xs),max(ys)-min(ys)]
                segmentation.append(f'{cls} '+' '.join(f'{v:.6f}' for v in a))
            else: box=a
            detection.append(f'{cls} '+' '.join(f'{v:.6f}' for v in box))
        (OUT/'labels_detect'/f'{p.stem}.txt').write_text('\n'.join(detection)+'\n')
        if len(segmentation)==len(shapes):
            (OUT/'labels_segment'/f'{p.stem}.txt').write_text('\n'.join(segmentation)+'\n')
    with LOG.open('a',newline='') as f:
        writer=csv.writer(f)
        if f.tell()==0: writer.writerow(['image','status','shapes'])
        writer.writerow([p.name,status,len(shapes)])
    print(f'{i}/{N_IMAGES} — {status}: {p.name}')

print('📁 Saved previews:',OUT/'annotated_previews')
print('📁 Your work is in your own Google Drive. Share the folder with the instructor.')
