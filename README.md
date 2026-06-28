# stereo-mosaic-live

[ one-line tagline — Tamir voice ]

[ hero media — interactive viewer GIF, added with the viewer plan ]

## What it does

[ short factual paragraph — Tamir voice ]

## Forked from

A continuation of [dynamic_panoramator](https://github.com/tamirelazar/dynamic_panoramator)
(faithful replication of Peleg, Ben-Ezra & Pritch 2001). This fork advances to the
Crossed-Slits projection (Zomet, Feldman, Peleg & Weinshall, PAMI 2003), made casual + live.

## Usage

```bash
smlive --input videos/boat.mp4 --out out --mode xslit --viewpoint 0.5
smlive --input my_clip.mp4 --out out --mode forward --stabilize --web-asset
```

Modes: `pushbroom` (the original), `xslit` (perspective-correct view), `forward` (synthesized forward motion).

## Method

[ short rigorous explainer — references PAMI 2003 + GLC ECCV 2004; Tamir voice ]

## Install

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv -r requirements.txt   # requires system ffmpeg
```
