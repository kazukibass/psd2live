# face3d-output — 顔パーツだけ 3D の頭で動かしたモデル

`moc3-cmo3-output/` と同じ PSD から作った別版。頭の向き（`ParamAngleX` / `ParamAngleY`）で、顔パーツ（顔の土台・白目・瞳・まつ毛・眉・鼻・口）だけを `tools/fit3d_head.py` の人の頭らしい 3D の頭（楕円体）で動かす。前髪・後ろ髪・耳は psd2live 標準の動きのまま。

| キー | 回転 |
| --- | --- |
| `ParamAngleX` −45 / +45 | 左 −20.0° / 右 +21.1° |
| `ParamAngleY` +30 / −30 | 上 +13.8° / 下 −14.4° |
| 四隅 | 左右の回転のあと上下の回転（組み合わせ） |

作り方:

```bash
(cd tools && python3 export_face3d.py <標準リグの geometry.json> ../reference/face_region/head3d_fit.json /tmp/face3d_disp.json)
tools/headless/build/install/psd2live-headless/bin/psd2live-headless --input psd-input/nekomimi.psd \
  --output face3d-output --atlas 4096 --mesh-spacing 40 --retarget-vertices /tmp/face3d_disp.json --render face3d-output/preview
```

`preview/` は姿勢シート、`harness_report.json` はハーネスの結果（顔の伸び縮みは 3D の回り込みとして基準より大きく、髪と耳のずれは psd2live 標準の動きから来るもの）。
