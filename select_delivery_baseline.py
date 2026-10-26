"""Compare existing local configurations on known development photos only."""
import json,hashlib,time,os
from collections import Counter
from datetime import datetime
from pathlib import Path
os.environ['YOLO_OFFLINE']='true'
os.environ['DO_NOT_TRACK']='1'
PROJECT=Path(__file__).resolve().parent
def main():
 import torch
 from ultralytics import YOLO
 torch.set_num_threads(8)
 expected=json.loads((PROJECT/'test_labels.json').read_text(encoding='utf-8'))
 expected.pop('04_multiple.jpg')
 expected['08_occlusion.jpg']={'counts':{'cup':1,'mouse':1,'scissors':1},'source':'user-confirmed earlier conversation'}
 expected['11_empty.jpg']={'counts':{},'source':'empty photograph supplied in conversation'}
 run=PROJECT/'outputs/delivery_baseline'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');run.mkdir(parents=True)
 report={'purpose':'known_development_samples_not_independent_acceptance','excluded':['04_multiple.jpg'],'configurations':[],'new_API_calls':0}
 for weights,agnostic in [('yolo11n.pt',False),('yolo11n.pt',True),('yolo11s.pt',True)]:
  model=YOLO(str(PROJECT/weights));rows=[]
  for name,truth in expected.items():
   path=PROJECT/'test_images'/name;start=time.perf_counter()
   result=model.predict(str(path),device='cpu',imgsz=1280,conf=.25,iou=.7,agnostic_nms=agnostic,verbose=False)[0]
   counts=dict(Counter(result.names[int(b.cls.item())] for b in result.boxes))
   err=sum(abs(counts.get(k,0)-truth['counts'].get(k,0)) for k in set(counts)|set(truth['counts']))
   rows.append({'image':name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'expected':truth['counts'],'counts':counts,'exact_match':counts==truth['counts'],'count_absolute_error':err,'detection_wall_seconds':time.perf_counter()-start})
  entry={'weights':weights,'agnostic_nms':agnostic,'conf':.25,'iou':.7,'imgsz':1280,'exact_count_matches':sum(r['exact_match'] for r in rows),'total_count_absolute_error':sum(r['count_absolute_error'] for r in rows),'rows':rows}
  report['configurations'].append(entry);print(json.dumps(entry,ensure_ascii=False),flush=True)
 report['warning']='Selection on known failures is development tuning, not a claim of general accuracy. Detection time excludes SAM.'
 (run/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('RESULT',run,flush=True)
if __name__=='__main__':main()
