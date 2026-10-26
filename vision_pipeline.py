"""Fixed local YOLO+SAM pipeline; no cloud requests or silent weight downloads."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parent
MODEL_FILE = 'yolo11s.pt'
PREDICT_OPTIONS = dict(device='cpu', imgsz=1280, conf=0.25, iou=0.7, agnostic_nms=True)
_lock = threading.Lock()
_models = None


def normalized_image(image):
    return ImageOps.exif_transpose(image).convert('RGB')


def image_identity(image):
    rgb = normalized_image(image)
    return hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode()+rgb.tobytes()).hexdigest()


def save_record(record, kind, images=None):
    folder = PROJECT/'outputs'/'app_runs'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+kind)
    folder.mkdir(parents=True, exist_ok=False)
    for name, image in (images or {}).items():
        if image is not None:
            (Image.fromarray(image) if isinstance(image, np.ndarray) else image).save(folder/name)
    path = folder/'result.json'
    record['result_path'] = str(path)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(path)


def run_local(image):
    global _models
    if image is None:
        raise ValueError('No input image')
    # Ultralytics predictors are mutable; serialize access across Gradio sessions.
    with _lock:
        loaded = 0.0
        if _models is None:
            for name in (MODEL_FILE, 'sam_b.pt'):
                if not (PROJECT/name).is_file():
                    raise FileNotFoundError(f'Missing local weight: {name}')
            os.environ['YOLO_OFFLINE'] = 'true'
            os.environ['DO_NOT_TRACK'] = '1'
            import torch
            from ultralytics import YOLO, SAM
            torch.set_num_threads(8)
            began = time.perf_counter()
            _models = (YOLO(str(PROJECT/MODEL_FILE)), SAM(str(PROJECT/'sam_b.pt')))
            loaded = time.perf_counter()-began
        start = time.perf_counter()
        rgb = normalized_image(image)
        identity = image_identity(rgb)
        bgr = np.asarray(rgb)[:, :, ::-1].copy()
        detector, segmenter = _models
        result = detector.predict(bgr, **PREDICT_OPTIONS, verbose=False)[0]
        counts = dict(Counter(result.names[int(b.cls.item())] for b in result.boxes))
        boxes = result.boxes.xyxy.cpu().tolist()
        annotated = result.plot()[:, :, ::-1].copy()
        segmented = None
        mask_count = 0
        if boxes:
            segmentation = segmenter.predict(bgr, bboxes=boxes, device='cpu', verbose=False)[0]
            segmented = segmentation.plot()[:, :, ::-1].copy()
            mask_count = len(segmentation.masks) if segmentation.masks is not None else 0
        elapsed = time.perf_counter()-start
        record = {'status':'success','scheme':'local_YOLO_SAM','image_sha256':identity,
                  'image_size':[rgb.width,rgb.height],'counts':counts,'model':MODEL_FILE,
                  'segmentation_model':'sam_b.pt','parameters':PREDICT_OPTIONS,
                  'processing_seconds':elapsed,'model_loading_seconds':loaded,
                  'timing_basis':'RGB normalization + detection + counting + SAM + rendering; excludes model loading, queue wait, file writes and browser rendering; first inference initialization may be included',
                  'mask_count':mask_count,'segmentation_skipped':not bool(boxes),
                  'objects':[{'name':result.names[int(b.cls.item())],'class_id':int(b.cls.item()),
                              'confidence':float(b.conf.item()),'xyxy':b.xyxy[0].tolist()} for b in result.boxes],
                  'counts_verified':False,'new_API_calls':0}
        path = save_record(record,'local',{'input.png':rgb,'detected.png':annotated,'segmented.png':segmented})
        return annotated, segmented, record, path


def compare_records(image, local, cloud):
    if image is None or local is None or cloud is None:
        return [], '请先完成当前照片的本地检测和大模型识别。'
    identity = image_identity(image)
    if any(r.get('status')!='success' or r.get('image_sha256')!=identity for r in (local,cloud)):
        return [], '结果不属于当前照片，请重新识别后比较。'
    a,b = local['counts'],cloud['counts']
    rows = [[k,a.get(k,0),b.get(k,0),'一致' if a.get(k,0)==b.get(k,0) else '不一致'] for k in sorted(set(a)|set(b))]
    message = '计数全部一致，但不代表识别正确，请结合原图核对。' if a==b else '存在计数差异，请检查漏检、误检或类别名称差异。'
    if not rows: message='两种方案均返回 0 件物品，请结合原图核对。'
    return rows,message
