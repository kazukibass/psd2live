# nekomimi — 1枚絵からゼロで作った Live2D モデル

正面の立ち絵 1 枚（`reference/front.png`、背景透過 1800×1600）だけから、psd2live 用のレイヤー分け PSD を作り、psd2live で `.moc3` / `.cmo3` まで書き出した例。`reference/angles-9dir.png`（頭の 9 方向シート）は、回転がどう見えるべきかの目標として使った。

## フォルダ

| パス | 中身 |
| --- | --- |
| `reference/front.png` | 入力の立ち絵 |
| `reference/angles-9dir.png` | 目標にした 9 方向の顔向きシート |
| `reference/poses_head.png` | 生成モデルの 9 方向（`ParamAngleX` ±30 × `ParamAngleY` ±30） |
| `reference/poses_face.png` | 口の開閉・まばたき・視線・首かしげ・体・呼吸 |
| `reference/poses_body.png` | 上半身での正面・体の向き・振り向き |
| `psd-input/nekomimi.psd` | 生成した 18 レイヤーの PSD（RGB 8bit、RLE 圧縮） |
| `moc3-cmo3-output/` | psd2live の出力（`.moc3`、`.cmo3`、physics、idle/blink/nod/shake モーション、4096 テクスチャ） |
| `tools/build_layers.py` | 立ち絵 → レイヤー分け PSD の生成スクリプト |
| `tools/headless/` | GUI（Compose）なしで psd2live のエンジンを動かす CLI と姿勢描画ツール |

## レイヤー構成（下 → 上）

`back hair` / `ears-r` / `ears-l` / `neck` / `topwear` / `ponytail` / `face` / `eyewhite-r` / `eyewhite-l` / `irides-r` / `irides-l` / `eyelash-r` / `eyelash-l` / `eyebrow-r` / `eyebrow-l` / `nose` / `mouth` / `front hair`

左右はキャラクター自身の左右（`-r` = 画面左）。

## どう分けたか

- **肌**: 線画（暗い画素）で区切られた領域の連結成分から、頬の点を含む成分を顔の肌とした。
- **目・眉・耳・ポニーテール**: 拡大画像で座標を測った多角形・楕円で切り出した。瞳はまつ毛の下に隠れていた上部を周囲の色で補完し、完全な楕円にした。
- **隠れていた部分の補完**: 顔の髪の下・目の下・首の上部・耳の付け根・服のポニーテールの下は、周囲の色で塗り足した（OpenCV の inpaint か中央値の単色）。
- **口**: 原画は閉じた一本線だけ。psd2live は口を「最大に開いた絵」として受け取り、中心線へ圧縮して閉じるため、開いた口（唇・口内・舌）を描き起こした。閉じると原画に近い一本線になる。
- **下まぶた**: 閉眼時に目と一緒に動くよう、白目レイヤーの下端に含めた（まつ毛レイヤーは仕様どおり上まつ毛だけ）。

## 再現手順

```bash
# 1. 立ち絵 → PSD（Python 3 + pillow numpy opencv-python-headless psd-tools）
python3 tools/build_layers.py reference/front.png psd-input/nekomimi.psd

# 2. GUI なし CLI をビルド（JDK 21）
../../gradlew -p tools/headless installDist

# 3. PSD → Live2D。--render を付けると reference/ に姿勢シートも出す
tools/headless/build/install/psd2live-headless/bin/psd2live-headless \
  --input psd-input/nekomimi.psd --output moc3-cmo3-output \
  --atlas 4096 --mesh-spacing 40 --render reference
```

Windows のデスクトップアプリで `psd-input/nekomimi.psd` を開いても、同じ構成で生成できる。

`tools/headless/` は、本体のビルドに必要な Google Maven（`dl.google.com`）へ届かない環境でも動くよう、Compose UI を除いた engine（`core` / `i18n` / `org.umamo`）と `ui/RigCanvasSupport.kt`（AWT のみ）だけをコンパイルする。

## 既知の限界

- 1 枚絵からの自動分割なので、隠れていた部分（髪の裏、耳の付け根、顔の輪郭の裏）は単色か周囲の色で埋めてあるだけ。大きく動かすと、塗り足しが平坦に見える場所がある。
- `ParamAngleZ` を 20° 以上にすると、ポニーテールの付け根で前髪とポニーテールの境目が少し見える。
- ポニーテール左端の細い毛先と、目尻のごく細い部分は、メッシュ化の際に 1〜5px 程度落ちる（psd2live の検証警告に出る）。
- 口の中・舌・閉じたときの線は描き起こしで、原画の作者が描いたものではない。
- 9 方向シートは目標として目視で比べただけで、キーフォームを 9 方向シートへ合わせる調整はしていない。
