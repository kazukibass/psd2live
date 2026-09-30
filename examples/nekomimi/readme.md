# nekomimi — 1枚絵からゼロで作った Live2D モデル

正面の立ち絵 1 枚（`reference/front.png`、背景透過 1800×1600）だけから、psd2live 用のレイヤー分け PSD を作り、psd2live で `.moc3` / `.cmo3` まで書き出した例。頭の向き（`ParamAngleX` / `ParamAngleY`）は、`reference/angles-9dir.png`（頭の 9 方向シート）から測った形に合わせてある。

## フォルダ

| パス | 中身 |
| --- | --- |
| `reference/front.png` | 入力の立ち絵 |
| `reference/angles-9dir.png` | 目標にした 9 方向の顔向きシート |
| `reference/poses_head.png` | 生成モデルの 9 方向（キーの端：`ParamAngleX` ±45 × `ParamAngleY` ±30） |
| `reference/compare_9dir.png` | 9 方向シート（上）と生成モデル（下）の方向ごとの比較 |
| `reference/poses_face.png` | 口の開閉・まばたき・視線・首かしげ・体・呼吸 |
| `reference/poses_body.png` | 上半身での正面・体の向き・振り向き |
| `psd-input/nekomimi.psd` | 生成した 18 レイヤーの PSD（RGB 8bit、RLE 圧縮） |
| `moc3-cmo3-output/` | psd2live の出力（`.moc3`、`.cmo3`、physics、idle/blink/nod/shake モーション、4096 テクスチャ） |
| `tools/build_layers.py` | 立ち絵 → レイヤー分け PSD の生成スクリプト |
| `tools/angle_targets.py` | 9 方向シートから測った目標位置と、方向ごとの変位場の生成 |
| `tools/headless/` | GUI（Compose）なしで psd2live のエンジンを動かす CLI、頭の向きの差し替え（`Retarget.kt`）、姿勢描画ツール |

## レイヤー構成（下 → 上）

`back hair` / `ears-r` / `ears-l` / `neck` / `topwear` / `ponytail` / `face` / `eyewhite-r` / `eyewhite-l` / `irides-r` / `irides-l` / `eyelash-r` / `eyelash-l` / `eyebrow-r` / `eyebrow-l` / `nose` / `mouth` / `front hair`

左右はキャラクター自身の左右（`-r` = 画面左）。

## どう分けたか

- **肌**: 線画（暗い画素）で区切られた領域の連結成分から、頬の点を含む成分を顔の肌とした。
- **顔の土台（`face`）**: 見えている顎の線画から顎先 (900,771) と顎のラインを測り、髪に隠れた頬と額は目尻の外側 (x 689 / 1111) を通る左右対称のアニメ顔として延長して、なめらかな輪郭にした。色は見えている肌の芯だけを残し、残りは肌から塗り足した。頬が髪の外へ出たときのために、下半分の輪郭に細い線を入れてある。
- **目・眉・耳・ポニーテール**: 拡大画像で座標を測った多角形・楕円で切り出した。瞳はまつ毛の下に隠れていた上部を周囲の色で補完し、完全な楕円にした。
- **隠れていた部分の補完**: 顔の髪の下・目の下・首の上部・耳の付け根・服のポニーテールの下は、周囲の色で塗り足した（OpenCV の inpaint か中央値の単色）。
- **口**: 原画は閉じた一本線だけ。psd2live は口を「最大に開いた絵」として受け取り、中心線へ圧縮して閉じるため、開いた口（唇・口内・舌）を描き起こした。閉じると原画に近い一本線になる。
- **下まぶた**: 閉眼時に目と一緒に動くよう、白目レイヤーの下端に含めた（まつ毛レイヤーは仕様どおり上まつ毛だけ）。
- **首とあごの裏**: 上を向くと顎の裏が見えるため、首レイヤーは上端を顎の幅まで広げ、上ほど暗くなる影を付けた。正面では顔の後ろに隠れている。

## 頭の向き（9 方向）

psd2live が自動生成する 9 軸の変形は、顔の輪郭をほとんど変えないため、9 方向シートと顔の土台の形が合わなかった。そこで次のように差し替えている。

1. **目標を測る**（`tools/angle_targets.py`）: 9 方向シートの各コマで、目・目尻・鼻・口・顎先・顎の左右・頬・眉間・額・頭頂・耳・髪の外側を、首輪のバックルを基準に測った。正面コマとの差を、目の間隔（モデル 239px ／ シート 87px）で拡大してモデルの座標へ移す。左右のコマは平均して左右対称にし、斜めは左右の差と上下の差を合成する（上下分は 0.7 倍。シートの斜めコマの実測に合わせた値）。
2. **変位場にする**: 測った点と、動かない胴体の点を通る thin-plate spline で、方向ごとになめらかな変位場を作る。
3. **キーフォームを差し替える**（`tools/headless/cli/Retarget.kt`）: 頭の下にあるデフォーマの `ParamAngleX/Y` キーフォームをすべて静止形にしてから、頭のメッシュ（顔・目・眉・鼻・口・耳・前髪・後ろ髪）の頂点に、各キー（X ±45 × Y ±30 の 8 方向）で変位場を足したキーフォームを入れる。既存の目の開閉・口の形と開き・眉・瞳の形のキーフォームとはすべて組み合わせて作るので、どの向きでもまばたきや口パクが動く。psd2live の `rigEdits` として渡すので、`.moc3` と `.cmo3` の両方に入る。

測った主な動き（シート上の px。モデルでは約 2.75 倍）:

| 方向 | 目 | 鼻 | 口 | 顎先 | 頭頂 | 耳 |
| --- | --- | --- | --- | --- | --- | --- |
| 上 | ↑40 | ↑64 | ↑53 | ↑44 | ↓5 | ↓53、外へ |
| 下 | ↓33 | ↓31 | ↓28 | ↓11（首輪にかかる） | ↑8 | ↓8、外へ |
| 右（左は鏡像） | →61 / →41、↑14 | →73 | →59 | →48 | →15 | 奥 →28、手前 ←11 |

## 再現手順

```bash
# 1. 立ち絵 → PSD（Python 3 + pillow numpy opencv-python-headless psd-tools）
python3 tools/build_layers.py reference/front.png psd-input/nekomimi.psd

# 2. GUI なし CLI をビルド（JDK 21）
../../gradlew -p tools/headless installDist

# 3. 9 方向シートから頭の向きの変位場を作る（Python 3 + numpy scipy）
python3 tools/angle_targets.py /tmp/angle_fields.json

# 4. PSD → Live2D。--retarget で頭の向きを差し替え、--render で reference/ に姿勢シートを出す
tools/headless/build/install/psd2live-headless/bin/psd2live-headless \
  --input psd-input/nekomimi.psd --output moc3-cmo3-output \
  --atlas 4096 --mesh-spacing 40 --retarget /tmp/angle_fields.json --render reference
```

Windows のデスクトップアプリで `psd-input/nekomimi.psd` を開くと、頭の向きは psd2live 標準の変形になる（手順 4 の差し替えは含まれない）。

`tools/headless/` は、本体のビルドに必要な Google Maven（`dl.google.com`）へ届かない環境でも動くよう、Compose UI を除いた engine（`core` / `i18n` / `org.umamo`）と `ui/RigCanvasSupport.kt`（AWT のみ）だけをコンパイルする。

## 既知の限界

- 1 枚絵からの自動分割なので、隠れていた部分（髪の裏、耳の付け根、顔の輪郭の裏）は単色か周囲の色で埋めてあるだけ。大きく動かすと、塗り足しが平坦に見える場所がある。
- `ParamAngleZ` を 20° 以上にすると、ポニーテールの付け根で前髪とポニーテールの境目が少し見える。
- ポニーテール左端の細い毛先と、目尻のごく細い部分は、メッシュ化の際に 1〜5px 程度落ちる（psd2live の検証警告に出る）。
- 口の中・舌・閉じたときの線は描き起こしで、原画の作者が描いたものではない。
- 9 方向シートの目印は手で測ったので、数 px 単位のずれはある。変形は 2D の変位で、髪の重なり順が向きで入れ替わるような表現（奥の髪が顔の後ろへ回るなど）はしていない。
- 横を向いたときの奥側の目は、シートほど細くならない。
- 頭の向きのキーフォームは頂点ごとに持つため、口のメッシュ（口の形 9 × 開き 33 の既存キー）で約 8,000 個になり、`.moc3` / `.cmo3` が大きめになる。
