#!/usr/bin/env python3
"""Genera las imágenes optimizadas del portafolio a partir de las fotos originales.

Uso (desde la carpeta «portafolio ximena»):
    python tools/build_gallery.py           # procesa solo las fotos nuevas o modificadas
    python tools/build_gallery.py --force   # vuelve a generar todo

Entradas
    ../CVXimena/fotos/**            fotos originales (no se modifican)
    ../CVXimena/FotoDeXimena.png    foto de perfil
    tools/gallery.config.json       colecciones, pies de foto ES/EN, destacadas y anexo del CV

Salidas
    assets/img/evidencias/<set>/<foto>-720.webp y -1600.webp
    assets/js/gallery-data.js       manifiesto que leen la galería y los carruseles
    assets/img/foto-*.jpg           recortes de la foto de perfil
    ../CVXimena/img/*.jpg           foto del CV y fotos del anexo de evidencias
    cv/*.pdf                        copias de los CV compilados

Cada foto se gira según su EXIF, se convierte de Display P3 a sRGB y se guarda sin
metadatos: las fotos de iPhone traen la ubicación GPS de donde se tomaron.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import re
import shutil
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor

from PIL import Image, ImageCms, ImageOps

try:  # optional: lets Pillow open .heic photos straight from an iPhone
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
OUT_DIR = os.path.join(SITE, 'assets', 'img', 'evidencias')
MANIFEST = os.path.join(SITE, 'assets', 'js', 'gallery-data.js')
SIZES = (720, 1600)                 # long edge: grid thumbnail, lightbox/carousel image
QUALITY = {720: 72, 1600: 80}
EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif')
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))


# ─── names ───────────────────────────────────────────────────────────────────

def fix_mojibake(name):
    """Undo zip mojibake: macOS UTF-8 names read as CP437 ('cerveceri╠üa' → 'cervecería')."""
    try:
        name = name.encode('cp437').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return unicodedata.normalize('NFC', name)


def slug(text):
    text = unicodedata.normalize('NFKD', fix_mojibake(text))
    text = ''.join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r'[^a-z0-9]+', '-', text).strip('-')


def folder_key(path):
    return '/'.join(slug(p) for p in re.split(r'[\\/]', path) if p and p != '.')


def photo_slug(stem):
    s = slug(stem)
    return s[:8] if len(s) > 20 else s      # camera UUID names → first 8 characters


# ─── images ──────────────────────────────────────────────────────────────────

def load(path, long_edge=None):
    """Open a photo upright, in sRGB, without metadata."""
    im = Image.open(path)
    if long_edge and im.format == 'JPEG':
        # Let the JPEG decoder downscale by 1/2, 1/4… while keeping the long edge ≥ long_edge
        k = long_edge / max(im.size)
        if k < 1:
            im.draft('RGB', (int(im.width * k) + 1, int(im.height * k) + 1))
    icc = im.info.get('icc_profile')
    im = ImageOps.exif_transpose(im)
    im = im.convert('RGB')
    if icc:
        try:
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)), SRGB, outputMode='RGB')
        except (ImageCms.PyCMSError, OSError):
            pass
    return im


def avg_color(im):
    r, g, b = im.convert('RGB').resize((1, 1), Image.BOX).getpixel((0, 0))
    return f'#{r:02x}{g:02x}{b:02x}'


def exif_date(path):
    try:
        ex = Image.open(path).getexif()
        d = ex.get_ifd(0x8769).get(36867) or ex.get(306) or ''
        return str(d).replace(':', '-', 2)[:10]
    except Exception:
        return ''


def process(job):
    """Worker: write the WebP sizes of one photo (skipped when up to date)."""
    src, base, force = job
    outs = {s: f'{base}-{s}.webp' for s in SIZES}
    mtime = os.path.getmtime(src)
    if not force and all(os.path.exists(p) and os.path.getmtime(p) >= mtime for p in outs.values()):
        dims = {s: Image.open(p).size for s, p in outs.items()}
        return dims, avg_color(Image.open(outs[min(SIZES)])), False
    im = load(src, max(SIZES))
    os.makedirs(os.path.dirname(base), exist_ok=True)
    dims = {}
    for s in sorted(SIZES, reverse=True):
        im.thumbnail((s, s), Image.LANCZOS)          # only ever shrinks
        im.save(outs[s], 'WEBP', quality=QUALITY[s], method=6)
        dims[s] = im.size
    return dims, avg_color(im), True


def square(im, focus=0.5):
    w, h = im.size
    s = min(w, h)
    if h > w:
        top = round((h - s) * focus)
        return im.crop((0, top, s, top + s))
    left = round((w - s) * focus)
    return im.crop((left, 0, left + s, s))


def save_jpeg(im, path, quality=85):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, 'JPEG', quality=quality, optimize=True, progressive=True)


def site_path(rel):
    return os.path.normpath(os.path.join(SITE, rel))


# ─── build ───────────────────────────────────────────────────────────────────

def discover(root):
    """{folder key: (dir, [files])} for every folder under root that holds photos."""
    found = {}
    for dp, dn, fn in os.walk(root):
        dn.sort()
        files = sorted(f for f in fn if f.lower().endswith(EXTS) and not f.startswith('.'))
        if files:
            found[folder_key(os.path.relpath(dp, root))] = (dp, files)
    return found


def plan_sets(cfg, found):
    """Match configured sets to folders; unknown folders become sets of their own."""
    sets, claimed = [], set()
    for s in cfg['sets']:
        key = folder_key(s['folder'])
        if key not in found:
            print(f'  ! No existe la carpeta «{s["folder"]}» (set {s["id"]}); se omite.')
            continue
        claimed.add(key)
        sets.append(dict(s, dir=found[key][0], files=found[key][1]))
    other = cfg['other']
    for key, (dp, files) in found.items():
        if key in claimed:
            continue
        name = fix_mojibake(os.path.basename(dp))
        title = name[:1].upper() + name[1:]
        print(f'  + Carpeta nueva sin configurar: «{key}» → colección «{other["es"]}»')
        sets.append({'id': key.replace('/', '-'), 'collection': other['id'], 'es': title, 'en': title,
                     'caption': {'es': title, 'en': title}, 'photos': {}, 'dir': dp, 'files': files})
    return sets


def ordered_photos(s):
    """[(file, slug, config)] in display order: config order first, then the rest by date."""
    conf = {k.lower(): v for k, v in s.get('photos', {}).items()}
    order = [k.lower() for k in s.get('photos', {})]
    excluded = {e.lower() for e in s.get('exclude', [])}
    by_stem = {}
    for f in s['files']:
        stem = os.path.splitext(f)[0].lower()
        if stem not in excluded:
            by_stem[stem] = f
    for k in conf:
        if k not in by_stem and k not in excluded:
            print(f'  ! {s["id"]}: «{k}» está en la configuración pero no en la carpeta.')
    listed = [k for k in order if k in by_stem]
    rest = sorted((k for k in by_stem if k not in conf),
                  key=lambda k: (exif_date(os.path.join(s['dir'], by_stem[k])), k))
    return [(by_stem[k], photo_slug(os.path.splitext(by_stem[k])[0]), conf.get(k, {})) for k in listed + rest]


def write_manifest(data):
    dumps = lambda v: json.dumps(v, ensure_ascii=False, separators=(',', ':'))
    lines = ['/* Generado por tools/build_gallery.py — no editar a mano. */', 'window.GALLERY = {']
    for key in ('base', 'sizes', 'total', 'featured', 'collections', 'sets'):
        lines.append(f' {dumps(key)}: {dumps(data[key])},')
    lines.append(' "photos": {')
    lines.append(',\n'.join(f'  {dumps(k)}: {dumps(v)}' for k, v in data['photos'].items()))
    lines.append(' }')
    lines.append('};')
    text = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    stamp_pages(hashlib.sha1(text.encode('utf-8')).hexdigest()[:8])


def stamp_pages(version):
    """Point every page at the new manifest (gallery-data.js?v=…) so browsers don't keep a cached copy."""
    for page in glob.glob(os.path.join(SITE, '**', '*.html'), recursive=True):
        with open(page, encoding='utf-8') as f:
            html = f.read()
        new = re.sub(r'(gallery-data\.js)\?v=[\w.-]*', rf'\g<1>?v={version}', html)
        if new != html:
            with open(page, 'w', encoding='utf-8', newline='\n') as f:
                f.write(new)


def remove_stale(expected):
    removed = 0
    for dp, dn, fn in os.walk(OUT_DIR, topdown=False):
        for f in fn:
            p = os.path.normpath(os.path.join(dp, f))
            if f.endswith('.webp') and p not in expected:
                os.remove(p)
                removed += 1
        if dp != OUT_DIR and not os.listdir(dp):
            os.rmdir(dp)
    return removed


def build_crops(cfg, force):
    for c in cfg.get('crops', []):
        src, out = site_path(c['source']), site_path(c['out'])
        if not os.path.exists(src):
            print(f'  ! No existe {c["source"]}')
            continue
        if not force and os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
            continue
        im = load(src).crop(tuple(c['box'])).resize(tuple(c['size']), Image.LANCZOS)
        save_jpeg(im, out, 88)
        print(f'  - {c["out"]}')


def build_cv_annex(cfg, sources, force):
    annex = cfg.get('cv_annex')
    if not annex:
        return
    for name, spec in annex['photos'].items():
        spec = {'photo': spec} if isinstance(spec, str) else spec
        set_id, _, stem = spec['photo'].partition('/')
        src = sources.get(f'{set_id}/{photo_slug(stem)}')
        if not src:
            print(f'  ! Anexo CV: no se encontró {spec["photo"]}')
            continue
        out = os.path.join(site_path(annex['out']), f'{name}.jpg')
        if not force and os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
            continue
        im = square(load(src, annex['size'] * 2), spec.get('focus', 0.5))
        save_jpeg(im.resize((annex['size'],) * 2, Image.LANCZOS), out, 85)
        print(f'  - CV: {name}.jpg')


def copy_pdfs(cfg):
    for src, dst in cfg.get('copy', {}).items():
        if src.startswith('_'):
            continue
        src, dst = site_path(src), site_path(dst)
        if os.path.exists(src) and (not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src)):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f'  - {os.path.relpath(dst, SITE)}')


def main():
    sys.stdout.reconfigure(errors='replace')
    ap = argparse.ArgumentParser(description='Optimiza las fotos de evidencias del portafolio.')
    ap.add_argument('--force', action='store_true', help='regenera todas las imágenes')
    args = ap.parse_args()

    with open(os.path.join(HERE, 'gallery.config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    root = site_path(cfg['source'])
    print(f'Fotos originales: {root}')
    sets = plan_sets(cfg, discover(root))

    jobs, meta, sources = [], [], {}
    for s in sets:
        for file, pslug, conf in ordered_photos(s):
            pid = f'{s["id"]}/{pslug}'
            src = os.path.join(s['dir'], file)
            sources[pid] = src
            jobs.append((src, os.path.join(OUT_DIR, s['id'], pslug), args.force))
            meta.append((s, pid, conf))

    print(f'Procesando {len(jobs)} fotos...')
    photos, fresh = {}, 0
    with ProcessPoolExecutor() as pool:
        for (s, pid, conf), (dims, color, made) in zip(meta, pool.map(process, jobs, chunksize=4)):
            (tw, th), (w, h) = dims[min(SIZES)], dims[max(SIZES)]
            cap = s.get('caption', {})
            photos[pid] = {'w': w, 'h': h, 'tw': tw, 'th': th, 'c': color,
                           'es': conf.get('es') or cap.get('es', ''), 'en': conf.get('en') or cap.get('en', '')}
            if made:
                fresh += 1
                print(f'  [{fresh}] {pid}')

    used = {s['collection'] for s in sets}
    cols = [c for c in cfg['collections'] if c['id'] in used]
    if cfg['other']['id'] in used:
        cols.append(cfg['other'])
    collections = [dict(c, sets=[s['id'] for s in sets if s['collection'] == c['id']]) for c in cols]
    featured = sorted((conf['featured'], pid) for s, pid, conf in meta if conf.get('featured'))
    write_manifest({
        'base': 'assets/img/evidencias/',
        'sizes': list(SIZES),
        'total': len(photos),
        'featured': [pid for _, pid in featured],
        'collections': collections,
        'sets': {s['id']: {'es': s['es'], 'en': s['en'], 'col': s['collection'],
                           'photos': [pid.split('/', 1)[1] for t, pid, _ in meta if t is s]} for s in sets},
        'photos': photos,
    })

    expected = {os.path.normpath(f'{base}-{size}.webp') for _, base, _ in jobs for size in SIZES}
    removed = remove_stale(expected)
    build_crops(cfg, args.force)
    build_cv_annex(cfg, sources, args.force)
    copy_pdfs(cfg)

    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(OUT_DIR) for f in fn)
    print(f'Listo: {len(photos)} fotos ({fresh} nuevas o actualizadas, {removed} archivos obsoletos borrados), '
          f'{total / 1048576:.1f} MB en assets/img/evidencias.')


if __name__ == '__main__':
    main()
