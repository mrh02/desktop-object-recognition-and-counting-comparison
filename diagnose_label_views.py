"""Bounded offline prompt/crop experiment. Outputs are NOT training labels."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'outputs' / 'auto_label_deps'))
os.environ.update(HF_HOME=str(PROJECT/'outputs'/'auto_label_models'),
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                  TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')


def iou(a, b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]))
    y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=x*y
    return inter / max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--files',nargs='+',default=['01.jpg','05.jpg','08.jpg','19.jpg','31.png'])
    args=parser.parse_args()
    if not args.execute:
        print('Offline experiment: max 5 training images, 7 views per image, no labels/training/API. Add --execute.')
        return
    if not 1 <= len(args.files) <= 5 or len(set(args.files)) != len(args.files):
        parser.error('Use 1-5 unique training image names')
    sources=[PROJECT/'training_photos'/name for name in args.files]
    for p in sources:
        if p.parent.resolve() != (PROJECT/'training_photos').resolve() or not p.is_file():
            parser.error('Input must be an existing direct child of training_photos')
    import torch
    from torchvision.ops import nms
    from PIL import Image,ImageOps,ImageDraw,ImageFont
    from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
    from auto_label_trial import PROMPTS,PROMPT_SETS,LABELS,build_prompt_spans,token_class_scores
    from refine_label_candidates import resolve_class
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    descriptive=tuple({'a cup':'a paper drinking cup','a computer mouse':'a wired computer mouse',
                       'a usb cable':'a computer cable'}.get(p,p) for p in PROMPT_SETS['desktop_v2'])
    LABELS.update({'paper drinking cup':('cup',41),'wired computer mouse':('mouse',64),'computer cable':('cable',None)})
    sets={'original':PROMPTS,'desktop_v2':PROMPT_SETS['desktop_v2'],'descriptive':descriptive}
    out=PROJECT/'outputs'/'label_view_diagnostic'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    print(f'Output: {out}',flush=True)
    model_id='IDEA-Research/grounding-dino-tiny'
    processor=AutoProcessor.from_pretrained(model_id,local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(model_id,local_files_only=True).eval()
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    report={'model':model_id,'purpose':'diagnostic_only_no_training_labels','labels_verified':False,
            'new_api_calls':0,'new_api_tokens':0,'training_started':False,
            'box_threshold':.35,'class_score':.35,'class_margin':.1,'per_view_agnostic_nms':.7,
            'tile_fraction':.65,'internal_edge_fraction':.025,'merge_same_class_iou':.5,
            'prompt_sets':sets,'images':[],'status':'running',
            'limitations':['Prompt scope is limited, not a complete COCO80 annotation system.',
                          'Same-model view agreement is not independent verification.',
                          'Unsupported/unknown objects and missing boxes can remain.',
                          'No filenames or expected counts enter inference or decoding.']}
    def preview(im,dets,title):
        p=im.copy(); p.thumbnail((480,560)); canvas=Image.new('RGB',(500,600),'white'); canvas.paste(p,(0,32))
        d=ImageDraw.Draw(canvas); d.text((5,5),title,fill='black',font=font)
        for det in dets:
            b=[v*(p.width/im.width if k%2==0 else p.height/im.height)+(32 if k%2 else 0) for k,v in enumerate(det['box_xyxy'])]
            color='#c02020' if det.get('touches_internal_edge') or det['name'] is None else '#006baf'
            d.rectangle(b,outline=color,width=2)
            label=f"{det['name'] or '?'} {det['score']:.2f}"
            pos=(max(0,min(350,b[0])),max(32,b[1]-18))
            d.rectangle(d.textbbox(pos,label,font=font),fill='white'); d.text(pos,label,fill=color,font=font)
        return canvas
    for source in sources:
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        with Image.open(source) as opened: im=ImageOps.exif_transpose(opened).convert('RGB')
        w,h=im.size; tw,th=round(w*.65),round(h*.65)
        views=[('full_'+key,key,(0,0,w,h)) for key in sets]
        views += [(f'tile_{i}','desktop_v2',b) for i,b in enumerate([(0,0,tw,th),(w-tw,0,w,th),(0,h-th,tw,h),(w-tw,h-th,w,h)])]
        row={'image':str(source),'source_sha256':source_hash,'width':w,'height':h,'views':[]}
        for name,key,region in views:
            started=time.perf_counter(); crop=im.crop(region); text='. '.join(sets[key])+'.'
            ids,spans=build_prompt_spans(processor.tokenizer,text,sets[key])
            if len(ids)>model.config.max_text_len: raise ValueError('Prompt too long')
            inputs=processor(images=crop,text=text,return_tensors='pt')
            if inputs.input_ids[0].tolist()!=ids: raise ValueError('Token alignment mismatch')
            with torch.inference_mode(): pred=model(**inputs)
            probs=pred.logits[0].sigmoid(); indices=(probs.max(-1).values>.35).nonzero(as_tuple=True)[0].tolist()
            raw=[]
            for idx in indices:
                scores=token_class_scores(probs[idx].tolist(),spans); resolved=resolve_class(scores,.35,.1)
                cx,cy,bw,bh=pred.pred_boxes[0,idx].tolist()
                local=[max(0,min(crop.width,(cx-bw/2)*crop.width)),max(0,min(crop.height,(cy-bh/2)*crop.height)),
                       max(0,min(crop.width,(cx+bw/2)*crop.width)),max(0,min(crop.height,(cy+bh/2)*crop.height))]
                if local[2]<=local[0] or local[3]<=local[1]: continue
                box=[v+region[k%2] for k,v in enumerate(local)]
                assert 0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h
                edge=((region[0]>0 and local[0]<.025*crop.width) or (region[1]>0 and local[1]<.025*crop.height)
                      or (region[2]<w and local[2]>.975*crop.width) or (region[3]<h and local[3]>.975*crop.height))
                raw.append({'query_index':idx,'name':resolved['name'],'coco_class_id':resolved['coco_class_id'],
                            'score':resolved['score'],'resolution':resolved,'prompt_scores':scores,'box_xyxy':box,
                            'detector_score':float(probs[idx].max()),'view':name,'touches_internal_edge':bool(edge)})
            keep=nms(torch.tensor([r['box_xyxy'] for r in raw]),torch.tensor([r['detector_score'] for r in raw]),.7).tolist() if raw else []
            kept=[raw[i] for i in keep]
            row['views'].append({'name':name,'prompt_set':key,'crop_xyxy':region,'seconds':time.perf_counter()-started,
                                 'counts':dict(Counter(r['name'] or 'unresolved' for r in kept)),'detections':kept,'raw_detections':raw})
            print(source.name,name,row['views'][-1]['counts'],f"{row['views'][-1]['seconds']:.1f}s",flush=True)
        # Union only within the unchanged desktop_v2 prompt condition. Keep class conflicts visible.
        proposals=[r for v in row['views'] if v['prompt_set']=='desktop_v2' for r in v['detections'] if not r['touches_internal_edge']]
        merged=[]
        for r in sorted(proposals,key=lambda x:x['score'],reverse=True):
            if not any(r['name']==s['name'] and iou(r['box_xyxy'],s['box_xyxy'])>=.5 for s in merged): merged.append(r)
        row['tile_union_diagnostic']=merged
        row['union_counts']=dict(Counter(r['name'] or 'unresolved' for r in merged))
        row['cross_class_overlaps']=[{'a':i,'b':j,'iou':iou(a['box_xyxy'],b['box_xyxy'])} for i,a in enumerate(merged) for j,b in enumerate(merged[:i]) if a['name']!=b['name'] and iou(a['box_xyxy'],b['box_xyxy'])>=.5]
        panels=[preview(im,v['detections'],v['name']) for v in row['views'][:3]]+[preview(im,merged,'desktop_v2 + tiles (NOT labels)')]
        sheet=Image.new('RGB',(2000,600),'white')
        for i,p in enumerate(panels): sheet.paste(p,(i*500,0))
        sheet.save(out/(source.stem+'_comparison.jpg'))
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
        report['images'].append(row)
        (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    report['status']='completed_diagnostic_not_accepted_labels'
    report['source_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'runner_source.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print('Completed. No training labels, cloud calls, or application changes.',flush=True)


if __name__=='__main__': main()
