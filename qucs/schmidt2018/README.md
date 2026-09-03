# Schmidt2018再現netlistをQucs-Sで実行する

## 提供ファイル

| ファイル | 用途 |
|---|---|
| `schmidt2018_reproduction_ngspice.cir` | 再現計算の最終反復で使った回路deckを安定した場所へ複製したもの。`waveforms.dat`を出力する。 |
| `schmidt2018_reproduction_qucs.cir` | 回路・プラズマ素子値は同一で、Qucs-S用の`spice4qucs.tr1.plot`も出力するもの。 |

固定されている自己無撞着状態は次の通りである。

| 項目 | 値 |
|---|---:|
| 電子温度 `Te` | 4.749293722676241 eV |
| 電子密度 `ne` | 1.245160090231506e15 m^-3 |
| 電源 | 100 V振幅、13.56 MHz、cos位相 |
| 整合素子 | `Cmatch1=1.55 nF`、`Cmatch2=175 pF`、`Lmatch=1.5 uH` |
| 過渡解析 | 300周期、240点/周期、末尾24周期を保存 |

## Qucs-Sでの実行

1. Qucs-Sのシミュレータ設定でバックエンドをngspiceにする。
2. ngspice実行ファイルに`C:\Spice64\bin\ngspice_con.exe`を指定する。
3. `schmidt2018_reproduction_qucs.cir`をQucs-SのSPICE netlistとして開き、過渡解析を実行する。
4. `spice4qucs.tr1.plot`から、`v(plasma)`、`v(match)`、`i(Vsense_generator)`などを表示する。

Qucs-Sが任意の`.cir`を直接実行できない構成の場合は、Qucs-Sで生成した`spice4qucs.cir`を閉じた状態で本ファイルの内容へ置き換え、同じngspiceバックエンドで実行する。元ファイルは先に別名保存すること。

PowerShellから同じdeckを直接確認する場合は、次を実行する。

```powershell
Set-Location "C:\Users\TK\Documents\ChatGPT\プラズマ等価回路\qucs\schmidt2018"
& "C:\Spice64\bin\ngspice_con.exe" -n -o qucs_run.log schmidt2018_reproduction_qucs.cir
```

## 観測量

| SPICEベクトル | 意味 |
|---|---|
| `v(src)` | 理想RF電源電圧 |
| `v(match)` | 50 ohm電源抵抗後の整合回路入力電圧 |
| `v(load_sense)` | 整合インダクタ後の負荷側電圧 |
| `v(plasma)` | powered electrodeのプラズマ端子電圧 |
| `v(bulk1)-v(plasma)` | powered sheath電圧 |
| `v(bulk2)` | grounded sheath電圧 |
| `i(Vsense_generator)` | 発振器電流 |
| `i(Vsense_load)` | 整合回路の直列電流 |
| `i(Vsense_stray)` | 寄生容量枝電流 |
| `i(Vsense_plasma)` | プラズマ枝電流 |

Qucs-S版では`v(bulk1)`と`v(plasma)`を別々に保存しているため、powered sheath電圧はデータ表示式で差を取る。

## 期待する基準値

末尾20周期の波形解析による再現基準値は次の通りである。

| 指標 | 基準値 |
|---|---:|
| 入力インピーダンス | 49.5535 + j0.3961 ohm |
| プラズマ電圧基本波振幅 | 355.315 V |
| プラズマ電圧DCオフセット | -244.533 V |
| プラズマ吸収電力 | 4.72468 W |
| 発振器電流基本波振幅 | 1.00446 A |
| 回路電力収支残差 | 0.0215% |

## 重要な制約

このdeckは、Pythonのグローバルモデル反復で得た最終`ne`と`Te`を固定した一回のRF回路計算である。Qucs-S単体で`ne`・`Te`を更新する自己無撞着反復は行わない。部品値を変更した後も自己無撞着解を得るには、Python側の`plasma-reproduce`を再実行してプラズマパラメータを更新する必要がある。
