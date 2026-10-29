"""Exercise the deployed local Gradio API. Never call the cloud endpoint."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

from gradio_client import Client, handle_file
from PIL import Image, ImageOps


def identity(path):
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        return hashlib.sha256(
            f'{rgb.width}x{rgb.height}:RGB:'.encode() + rgb.tobytes()
        ).hexdigest()


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7863')
    parser.add_argument('images', nargs='*', type=Path)
    args = parser.parse_args()
    images = args.images or [project / 'test_images' / name for name in (
        '05_multiple.jpg', '11_empty.jpg'
    )]
    for path in images:
        if not path.is_file():
            parser.error(f'Missing image: {path}')

    with urllib.request.urlopen(args.url, timeout=15) as response:
        assert response.status == 200
    client = Client(args.url, verbose=False)
    results = []
    for path in images:
        output = client.predict(handle_file(str(path.resolve())), api_name='/detect_image')
        if not output[-1]:
            raise RuntimeError(f'Local pipeline failed: {output[2]}')
        record = json.loads(Path(output[-1]).read_text(encoding='utf-8'))
        assert record['status'] == 'success'
        assert record['model'] == 'yolo11s.pt'
        assert record['segmentation_model'] == 'sam_b.pt'
        assert record['parameters']['device'] == 'cpu'
        assert record['image_sha256'] == identity(path)
        assert record['new_API_calls'] == 0
        assert record['counts'] == dict(Counter(obj['name'] for obj in record['objects']))
        assert output[0] and Path(output[0]).is_file()
        if record['objects']:
            assert not record['segmentation_skipped']
            assert record['mask_count'] > 0
            assert output[1] and Path(output[1]).is_file()
        else:
            assert record['segmentation_skipped'] and output[1] is None
        results.append({'image': path.name, 'record': record})
        print(f"VERIFIED {path.name}: {record['counts']}", flush=True)

    # A missing image must not reuse the preceding successful result.
    missing = client.predict(None, api_name='/detect_image')
    assert missing[0] is None and missing[1] is None and missing[-1] is None
    report = {
        'status': 'passed', 'url': args.url, 'results': results,
        'checks': ['HTTP page', 'real image upload', 'CPU YOLO+SAM',
                   'downloadable images and JSON', 'image content hash',
                   'missing input clears outputs'],
        'new_API_calls': 0,
        'limitations': 'Known development images; no independent accuracy acceptance or live cloud test.',
    }
    folder = project / 'outputs' / 'docker_checks' / datetime.now().strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True, exist_ok=False)
    target = folder / 'checks.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'PASSED. Report: {target}', flush=True)


if __name__ == '__main__':
    main()
