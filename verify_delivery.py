"""Local integration checks. Cloud response is mocked, never transmitted."""
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parent

def main():
 from PIL import Image
 import app
 from vision_pipeline import compare_records,image_identity
 from gradio_client import Client,handle_file
 import requests
 run=PROJECT/'outputs/delivery_baseline/20261023_195951_332346'
 checks=[]
 assert app.detect_image(None)[3] is None
 assert app.recognize_current(None)[1] is None
 checks.append('missing input clears result states')
 first=Image.open(PROJECT/'test_images/05_multiple.jpg')
 second=Image.open(PROJECT/'test_images/11_empty.jpg')
 base={'status':'success','image_sha256':image_identity(first),'counts':{'cup':1}}
 assert compare_records(second,base,base)[0]==[]
 assert compare_records(first,base,None)[0]==[]
 assert compare_records(first,base,base)[0]==[['cup',1,1,'一致']]
 checks.append('current image content hash checked before comparison')
 real_cloud,real_save=app.recognize_image,app.save_record
 captured=[]
 def mock_save(record,kind):
  record['test_mock']=True;captured.append(record);record['result_path']=str(run/'mock_not_a_real_API_result.json');return record['result_path']
 app.save_record=mock_save
 try:
  app.recognize_image=lambda im:('MOCK_RESPONSE_NO_API',{'cup':1})
  text,record,path=app.recognize_current(first)
  assert record['image_sha256']==image_identity(first) and 'MOCK' in text
  app.recognize_image=lambda im:('MOCK_FAILURE_NO_API',None)
  assert app.recognize_current(first)[1:] == (None,None)
  app.recognize_image=lambda im:(_ for _ in ()).throw(RuntimeError('mock'))
  assert app.recognize_current(first)[1:] == (None,None)
 finally:
  app.recognize_image,app.save_record=real_cloud,real_save
 checks.append('mock cloud success, invalid response and failure reset; zero API requests')
 local_results=[]
 try:
  app.demo.launch(server_name='127.0.0.1',server_port=7862,prevent_thread_lock=True,inbrowser=False,share=False,quiet=True,allowed_paths=app.APP_ALLOWED_PATHS)
  assert requests.get('http://127.0.0.1:7862',timeout=10).status_code==200
  client=Client('http://127.0.0.1:7862',verbose=False)
  for name in ['05_multiple.jpg','11_empty.jpg']:
   response=client.predict(handle_file(str(PROJECT/'test_images'/name)),api_name='/detect_image')
   # Gradio State is server-only and therefore omitted from HTTP return values.
   result_path=Path(response[-1]);record=json.loads(result_path.read_text(encoding='utf-8'))
   assert record['model']=='yolo11s.pt' and record['status']=='success'
   if name=='05_multiple.jpg':
    assert record['counts']=={'scissors':1,'cup':1,'mouse':1}
    assert record['mask_count']==3 and not record['segmentation_skipped']
    assert response[0] and response[1]
   else:
    assert record['counts']=={} and record['segmentation_skipped'] and response[1] is None
   local_results.append({'image':name,'record':record,'download_path':str(result_path)})
   print('VERIFIED',name,record['counts'],round(record['processing_seconds'],2),flush=True)
  checks.append('real Gradio HTTP upload, YOLO+SAM outputs and JSON download for populated and empty images')
 finally:
  app.demo.close()
 report={'status':'passed','checks':checks,'local_results':local_results,'new_API_calls':0,
         'cloud_test':'mock only, no live cloud regression performed','temporary_server_port':7862,
         'note':'Known development images, not independent accuracy acceptance.'}
 (run/'delivery_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('ALL DELIVERY CHECKS PASSED',flush=True)

if __name__=='__main__':main()
