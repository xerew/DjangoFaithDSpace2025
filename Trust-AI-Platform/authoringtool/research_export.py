"""Helpers shared by the research exports (export_item_bank, export_scenario_bank).

They turn activity HTML into plain text with numbered image markers and copy
the images next to the export, so the study can run without the database.
"""
import base64
import html
import os
import re
import shutil

from django.conf import settings
from django.utils.html import strip_tags

IMG_TAG = re.compile(r'<img\b[^>]*?\bsrc\s*=\s*["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)
# Teachers often type the option letter themselves ("A. ...", "b) ..."). The
# study shuffles options and adds its own letters, so remove theirs.
OPTION_LABEL = re.compile(r'^\(?[A-Za-zΑ-Ωα-ω][.)]\s+')


def plain(html_text):
    return ' '.join(html.unescape(strip_tags(html_text or '')).split())


def text_with_images(activity, out_dir):
    """Return (plain text with "[Image n]" markers, copied image paths, images not copied)."""
    images, skipped = [], 0
    text = activity.text or ''
    for index, match in enumerate(IMG_TAG.finditer(text), start=1):
        path = copy_image(match.group(1), f'{activity.id}_{index}', out_dir)
        if path:
            images.append(path)
        else:
            skipped += 1
    numbers = iter(range(1, 1000))
    text = IMG_TAG.sub(lambda _match: f' [Image {next(numbers)}] ', text)
    return plain(text), images, skipped


def option_record(activity, option, out_dir):
    image = None
    if option.image:
        image = copy_file(option.image.path, f'{activity.id}_option{option.id}', out_dir)
    return {
        'id': option.id, 'text': OPTION_LABEL.sub('', plain(option.text)),
        'is_correct': option.is_correct, 'image': image,
    }


def copy_image(src, stem, out_dir):
    if src.startswith('data:image/'):
        header, _, data = src.partition(',')
        extension = header.split('/')[1].split(';')[0] or 'png'
        relative = os.path.join('images', f'{stem}.{extension}')
        with open(os.path.join(out_dir, relative), 'wb') as handle:
            handle.write(base64.b64decode(data))
        return relative.replace(os.sep, '/')
    media_url = settings.MEDIA_URL
    path = re.sub(r'^https?://[^/]+', '', src)
    if path.startswith(media_url):
        return copy_file(os.path.join(settings.MEDIA_ROOT, path[len(media_url):]), stem, out_dir)
    return None  # external image: not copied, counted in the manifest


def copy_file(source, stem, out_dir):
    if not os.path.exists(source):
        return None
    extension = os.path.splitext(source)[1] or '.png'
    relative = os.path.join('images', f'{stem}{extension}')
    shutil.copyfile(source, os.path.join(out_dir, relative))
    return relative.replace(os.sep, '/')
