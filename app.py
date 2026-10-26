import os
import tempfile
import time

os.environ['GRADIO_TEMP_DIR'] = tempfile.mkdtemp(prefix='lingjing-gradio-')
import gradio as gr
from vlm_service import recognize_image
from vision_pipeline import PROJECT, run_local, image_identity, normalized_image, save_record, compare_records

APP_ALLOWED_PATHS = [str(PROJECT/'outputs'/'app_runs')]


def detect_image(image):
    if image is None:
        return None,None,'请先上传一张照片。',None,None
    try:
        annotated,segmented,record,path=run_local(image)
        lines=[f'{k}：{v}个' for k,v in record['counts'].items()]
        lines.append(f"总计：{sum(record['counts'].values())}件")
        lines.append(f"本地完整处理耗时：{record['processing_seconds']:.2f}秒（不含模型加载）")
        if record['model_loading_seconds']:
            lines.append(f"首次模型加载：{record['model_loading_seconds']:.2f}秒")
        if record['segmentation_skipped']:
            lines.append('没有检测框，本次跳过分割。')
        return annotated,segmented,'\n'.join(lines),record,path
    except Exception as error:
        return None,None,f'本地处理失败：{type(error).__name__}。请检查模型文件和内存。',None,None


def recognize_current(image):
    if image is None:
        return '请先上传一张照片。',None,None
    # Invoked only by the separate user-operated cloud button.
    try:
        start=time.perf_counter()
        rgb=normalized_image(image)
        identity=image_identity(rgb)
        text,counts=recognize_image(rgb)
        elapsed=time.perf_counter()-start
        if counts is None:
            return text,None,None
        record={'status':'success','scheme':'cloud_VLM','model':'qwen3-vl-plus',
                'image_sha256':identity,'counts':counts,'processing_seconds':elapsed,
                'timing_basis':'normalization + encoding + client setup + full API request + parsing; excludes queue wait, file writes and browser rendering',
                'response_summary':text,'counts_verified':False}
        path=save_record(record,'cloud')
        return text+f'\n云端完整处理耗时：{elapsed:.2f}秒',record,path
    except Exception as error:
        return f'本次处理失败：{type(error).__name__}。',None,None


def compare_current(image,local,cloud):
    rows,message=compare_records(image,local,cloud)
    if image is None or local is None or cloud is None:
        return rows,message,None
    identity=image_identity(image)
    if any(r.get('status')!='success' or r.get('image_sha256')!=identity for r in (local,cloud)):
        return rows,message,None
    record={'status':'success','scheme':'comparison','image_sha256':identity,'rows':rows,
            'local_result':local['result_path'],'cloud_result':cloud['result_path'],
            'local_processing_seconds':local['processing_seconds'],'cloud_processing_seconds':cloud['processing_seconds'],
            'note':message,'independent_accuracy_evaluation':False,'new_API_calls':0}
    return rows,message,save_record(record,'comparison')


with gr.Blocks() as demo:
    local_state=gr.State(None)
    cloud_state=gr.State(None)
    gr.Markdown('# 桌面物品识别与清点对比系统')
    gr.Markdown('本地使用固定的 YOLO11s + SAM；大模型独立分析同一张照片。识别结果可能漏检或误判，请结合原图核对。')
    with gr.Row():
        input_image=gr.Image(label='上传照片',type='pil',sources=['upload'],format='png')
        detected=gr.Image(label='YOLO 检测结果',interactive=False,format='png')
    segmented=gr.Image(label='SAM 分割结果',interactive=False,format='png')
    detect_button=gr.Button('检测并分割')
    local_text=gr.Textbox(label='本地物品统计',lines=7,interactive=False)
    local_file=gr.File(label='下载本地结果 JSON',interactive=False)
    gr.Markdown('## 多模态大模型识别')
    gr.Markdown('仅点击下方按钮才会将当前照片发送至阿里云并消耗额度；不会自动审核或重复调用。')
    cloud_button=gr.Button('大模型识别')
    cloud_text=gr.Textbox(label='大模型物品统计',lines=8,interactive=False)
    cloud_file=gr.File(label='下载大模型结果 JSON',interactive=False)
    gr.Markdown('## 两种方案计数对比')
    compare_button=gr.Button('比较当前结果')
    table=gr.Dataframe(headers=['类别','YOLO 数量','大模型数量','是否一致'],datatype=['str','number','number','str'],value=[],interactive=False)
    comparison_text=gr.Textbox(label='对比说明',interactive=False)
    comparison_file=gr.File(label='下载对比 JSON',interactive=False)
    detect_button.click(fn=lambda:(None,None,'处理中…',None,None,[],'',None),inputs=None,
                        outputs=[detected,segmented,local_text,local_state,local_file,table,comparison_text,comparison_file],queue=False).then(
        detect_image,inputs=input_image,outputs=[detected,segmented,local_text,local_state,local_file],concurrency_limit=1)
    cloud_button.click(fn=lambda:('处理中…',None,None,[],'',None),inputs=None,
                       outputs=[cloud_text,cloud_state,cloud_file,table,comparison_text,comparison_file],queue=False).then(
        recognize_current,inputs=input_image,outputs=[cloud_text,cloud_state,cloud_file],concurrency_limit=1)
    compare_button.click(compare_current,inputs=[input_image,local_state,cloud_state],outputs=[table,comparison_text,comparison_file])
    input_image.change(fn=lambda:(None,None,'',None,None,'',None,None,[],'',None),inputs=None,
                       outputs=[detected,segmented,local_text,local_state,local_file,cloud_text,cloud_state,cloud_file,table,comparison_text,comparison_file],queue=False)

if __name__=='__main__':
    demo.launch(server_name='127.0.0.1',server_port=7861,inbrowser=True,share=False,allowed_paths=APP_ALLOWED_PATHS)
