# nekomimi — 1枚絵からゼロで作った Live2D モデル

正面の立ち絵 1 枚（`reference/front.png`、背景透過 1800×1600）だけから、psd2live 用のレイヤー分け PSD を作り、psd2live で `.moc3` / `.cmo3` まで書き出した例。頭の向き（`ParamAngleX` / `ParamAngleY`）は、形を崩さない数値基準（ハーネス）の範囲で、`reference/angles-9dir.png`（頭の 9 方向シート）に近づけてある。

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
| `tools/angle_targets.py` | 9 方向シートで測った目印の位置（正面からのずれ） |
| `tools/head3d.py` | 頭を 3D の立体とみなして回す変形。少数のパラメータを目印とハーネス制約に合わせて求める |
| `tools/harness.py` | 変形の数値チェック（ハーネス）。基準を外れた姿勢を不合格にする |
| `reference/head3d_fit.json` | 3D の当てはめ結果（パラメータと目印ごとの誤差） |
| `reference/harness_report.json` | 最終モデルのハーネス結果（姿勢・パーツごとの数値） |
| `tools/headless/` | GUI（Compose）なしで psd2live のエンジンを動かす CLI、頭の向きの差し替え（`Retarget.kt`）、形状の書き出し（`Export.kt`）、姿勢描画ツール |

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

### 経緯
psd2live 標準の 9 軸変形は顔の輪郭をほとんど変えない。最初の差し替え（目印ごとに位置を合わせ、間を補間する方法）は、目印の位置は合っても顔全体がゴムのように歪んだ。その反省から、今の方法は「形を崩さないこと」を先に数値で縛り、その範囲で参考シートに近づける。

### ハーネス（`tools/harness.py`）
各姿勢（キーの端 8 方向と、その中間 7 方向）で、正面との比較を数値で判定する。

| チェック | 中身 | 基準 |
| --- | --- | --- |
| 三角形の歪み | メッシュの三角形ごとの伸び方（縦横の伸び率の比）、面積比、裏返り | 顔まわり p95 ≤ 1.45、髪 p95 ≤ 1.6、裏返り 0 |
| パーツの曲がり | パーツ全体をアフィン変換（回転・縮み・傾き）で当てはめた残り | 顔まわり 2.5%、髪 5% 以内 |
| 目 | 白目の縦横比の変化、左右の目の大きさの比 | 変化 30% 以内、比 0.55 以上 |
| レイヤーのずれ | 正面で重なっている 後ろ髪↔前髪、耳↔前髪 が離れる量 | p95 12px 以内（超えると輪郭が二重に見える、耳が浮く） |

1 つでも外れた姿勢は不合格。最初の差し替え版は 16 姿勢すべて不合格（顔の三角形が一方向に最大約 3.2 倍に伸びていた）で、見た目の歪みを数値で捉えられることを確認してある。

### 変形（`tools/head3d.py`）
頭を 1 つのなめらかな立体（ドーム）とみなし、頭のメッシュの各頂点に奥行きを与えて、左右（yaw）・上下（pitch、首が支点）に回して正射影する。前髪・後ろ髪・耳は同じ髪の殻に載せ（後ろ髪は 8px、耳は 20px まで後ろにずらせる）、鼻先だけ前に出す。

決める値は中心・半径・曲がり具合・奥行き・支点・回転角など 15 個だけで、参考シートで測った目印（`tools/angle_targets.py`）に最小二乗で合わせる。そのとき上のハーネスの基準（少し厳しめ）を罰則として一緒に入れるので、「目印に近いが歪む」解は選ばれない。斜めは左右の回転のあとに上下の回転をかけるだけで、手で混ぜる係数はない。

求めた変位は `Retarget.kt` で頭のメッシュの頂点キーフォーム（`ParamAngleX` ±45 × `ParamAngleY` ±30）に入れる。psd2live の頭のデフォーマの角度キーフォームは静止形に置き換え、目の開閉・口・眉・瞳の既存キーフォームとはすべて組み合わせる。

当てはめ結果: 左右 17.8°、上 17.2°、下 12.6°。目印の平均誤差 24.7px（モデル座標。参考シート上で約 9px）。

## 再現手順

```bash
# 1. 立ち絵 → PSD（Python 3 + pillow numpy scipy opencv-python-headless psd-tools）
python3 tools/build_layers.py reference/front.png psd-input/nekomimi.psd

# 2. GUI なし CLI をビルド（JDK 21）
sh ../../gradlew -p tools/headless installDist
BIN=tools/headless/build/install/psd2live-headless/bin/psd2live-headless

# 3. psd2live 標準のリグを作り、静止形の形状を書き出す
$BIN --input psd-input/nekomimi.psd --output /tmp/base --export-geometry /tmp/geo_rest.json

# 4. 3D の当てはめ（ハーネス制約つき）→ 頂点ごとの変位
(cd tools && python3 head3d.py /tmp/geo_rest.json /tmp/disp.json --report ../reference/head3d_fit.json)

# 5. 変位を入れて書き出し、姿勢シートと形状を出す
$BIN --input psd-input/nekomimi.psd --output moc3-cmo3-output --atlas 4096 --mesh-spacing 40 \
  --retarget-vertices /tmp/disp.json --render reference --export-geometry /tmp/geo_final.json

# 6. ハーネス（不合格があれば終了コード 1）
python3 tools/harness.py /tmp/geo_final.json --report reference/harness_report.json
```

`tools/headless/` は、本体のビルドに必要な Google Maven（`dl.google.com`）へ届かない環境でも動くよう、Compose UI を除いた engine（`core` / `i18n` / `org.umamo`）と `ui/RigCanvasSupport.kt`（AWT のみ）だけをコンパイルする。

## 既知の限界

- **動きが参考シートより小さい**: 形を崩さない条件の中で合わせた結果、回転は左右約 18°・上下約 13〜17° に収まった。参考シートほど大きくは向かない。特に下向きは、顎が首輪にかかるところまで下がらない。正面 1 枚の絵を変形するだけでは、これ以上回すと歪みの基準を超える。大きく向かせるには、横顔寄りの目や輪郭などの差分パーツを描き足す必要がある。
- ハーネスは形の歪みとレイヤーのずれを見るが、「参考シートにどれだけ似ているか」の見た目の判定はしていない。似ているかどうかは `reference/compare_9dir.png` を人が見て判断する。
- 1 枚絵からの自動分割なので、隠れていた部分（髪の裏、耳の付け根、顔の輪郭の裏）は単色か周囲の色で埋めてあるだけ。
- ポニーテールの細い毛先が、斜め上・左向きで小さな点として離れて見える。上向きの斜めでは首の影が少し塊に見える。
- 口の中・舌・閉じたときの線は描き起こし。
- 頭の向きのキーフォームは頂点ごとに持つため、口のメッシュ（既存キー 297 個）で約 8,000 個になる。
- Windows のデスクトップアプリで PSD を直接開くと、psd2live 標準の変形になる（手順 3〜5 の差し替えは含まれない）。
