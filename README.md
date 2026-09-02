# プラズマ等価回路 - Schmidt et al. (2018) 再現

Schmidt, Mussenbrock, Trieschmann, *Plasma Sources Science and Technology* 27, 105017 (2018) の Ar-CCP、外部整合回路、グローバルモデル連成を、ngspice で再現するリポジトリです。

## 前提

- Windows 10/11
- Python 3.12 以上
- `uv`
- ngspice: `C:\Spice64\bin\ngspice_con.exe`

## セットアップと実行

```powershell
uv sync
uv run pytest
uv run plasma-reproduce --config configs/schmidt2018.json --output artifacts/schmidt2018/reproduction_strict
uv run plasma-validate --config configs/schmidt2018.json --base-summary artifacts/schmidt2018/reproduction_strict/summary.json --base-output artifacts/schmidt2018/reproduction_strict
uv run plasma-reproduce --config configs/schmidt2018.json --optimize-matching --output artifacts/schmidt2018/matched
uv run plasma-map --sweep configs/pressure_voltage_sweep.json --raw-output artifacts/pressure_voltage_map
uv run plasma-esc --config configs/esc_wafer_focus_ring.json --output artifacts/esc_wafer_focus_ring/baseline
uv run plasma-esc-sweep --sweep configs/esc_focus_capacitance_sweep.json
uv run plasma-esc-optimize --optimization configs/esc_matching_optimization.json
uv run plasma-esc-two-zone --config configs/esc_two_zone.json --output artifacts/esc_two_zone/validation
uv run plasma-esc-two-zone-coupled --config configs/esc_two_zone.json --output artifacts/esc_two_zone/self_consistent
uv run plasma-esc-uniformity-optimize --optimization configs/esc_uniformity_optimization.json
uv run plasma-qucs-one-zone --config configs/qucs_rlc_one_zone.json --output artifacts/qucs_rlc_one_zone/self_consistent
uv run plasma-qucs-match --search configs/qucs_rlc_matching_search.json
```

生成途中の netlist、波形、ログは `artifacts/` に保存されます。検証済みの集計値は `reports/data/`、図とレポートは `reports/` に保存します。結果の要約は [Schmidt2018 再現レポート](reports/Schmidt2018再現レポート.md)、係数の由来は [容量規約監査](reports/Capacitance規約監査.md)、標準点外の計算は [圧力―電源振幅予測マップ](reports/PressureVoltage予測マップ.md)、二表面ESCモデルは [Wafer―Focus Ringモデル](reports/ESC_Wafer_FocusRingモデル.md)、整合設計の教材は [二入力回路最適化レポート](reports/ESC_二入力回路最適化_教育レポート.md)、横結合の極限と保存則は [二ゾーンモデル検証](reports/ESC_二ゾーンモデル検証.md)、局所粒子・電力収支との反復連成は [二ゾーン自己無撞着モデル](reports/ESC_二ゾーン自己無撞着モデル.md)、Focus電気条件によるイオン束均一化は [二ゾーン均一性最適化](reports/ESC_二ゾーン均一性最適化.md)、Qucs-S回路の導入方法は [Qucs-S netlist連成手順](reports/Qucs-S_netlistグローバルモデル連成手順.md)、実装済みの最小構成は [Qucs-S RLC一ゾーン連成レポート](reports/Qucs-S_RLC一ゾーンプラズマ連成.md)、有限Qを含む整合条件は [Qucs-S RLC整合条件探索](reports/Qucs-S_RLC整合条件探索.md) を参照してください。

## 数値安定化

論文のシース容量 `C_s∝1/sqrt(V_s)` は `V_s=0` で発散し、電子電流 `I_e∝exp(-V_s/T_e)` は負のシース電圧で非物理的に増大します。本実装では次を使用します。

```text
Vabs_delta(V) = sqrt(V^2 + delta_C^2)
Vplus_delta(V) = (V + sqrt(V^2 + delta_e^2)) / 2
C_s,reg = alpha_C sqrt(K / Vabs_delta(V_s))
I_e,reg = I_e,sat exp(-Vplus_delta(V_s) / T_e)
```

これにより容量は有限となり、電子電流は滑らかに `0 < I_e <= I_e,sat` を満たします。既定の `delta_C=delta_e=0.05 V` は 1 V 以上で元式との差が十分小さく、感度試験で確認します。

`alpha_C=0.5862` は、公開された密度・整合容量で `Im(Z_in)` を合わせた標準点校正係数です。監査の結果、行列シースの物理的な微分容量は `alpha_C=0.5` であり、ngspice 27/28も現行版と同じ `I=C(V)dV/dt` 規約でした。`0.5` で再整合すると容量値は論文近傍になりますが、自己無撞着な密度と吸収電力は一致しません。このため `0.5862` を物理定数とはせず、標準点再現と範囲外予測で固定する経験係数として扱います。

## 標準点の結果

- `T_e = 4.7493 eV`、`n_e = 1.2452e15 m^-3`
- `Z_in = 49.55 + j0.40 ohm`
- プラズマ電圧振幅 `355.3 V`、DC オフセット `-244.5 V`
- `I_rf = 1.004 A`、`I_pl = 0.604 A`、`I_L = 6.659 A`、`I_stray = 6.055 A`
- 損失は電源抵抗 `25.22 W`、整合器 `11.08 W`、寄生枝 `9.18 W`、プラズマ `4.72 W`
- 全系電力収支残差 `0.021%`

## 圧力―電源振幅予測

論文の整合器を固定したまま、`0.5–10 Pa`、`50–200 V peak` の20条件を計算しました。全20条件が密度・RF周期の収束条件を満たし、計算例外は0件でした。圧力上昇に伴って固定回路の反射係数が最大 `0.896` まで悪化するため、広い運転範囲では整合器の再設計が必要です。

## Wafer―Focus Ring二表面ESCモデル

Wafer面とFocus-ring面を異なるESC容量を介して独立電極へ接続し、それぞれの非線形シース・バルク枝を共通グローバルプラズマへ連成しました。物理的な微分容量`alpha_C=0.5`を使用し、代表条件とFocus-ring容量`90–720 pF`の感度計算はすべて収束しています。

## 二入力整合回路の最適化

各入力へ並列Cと直列Lを追加し、解析的L型整合を初期値にngspiceで局所最適化しました。有限Qコイル、周期収束ペナルティ、自己無撞着な密度再検証を含みます。代表条件では二ポートの見かけの反射目的関数が`1.8487`から`0.9528`へ低下しました。一方でFocus-ring枝の逆潮流とシース電圧上昇も生じるため、整合だけを目的にした解を実機設定とはみなしません。

## Wafer―Focus Ring二ゾーン検証

Wafer側とFocus-ring側に独立した局所bulk節点と接地シースを置き、両者を電子運動量式に基づく有限の横方向R–L枝で接続しました。強結合端ではbulk電位差が平均RF振幅の`0.0195%`まで低下し、閉鎖輸送試験では全粒子数と電子エネルギーを保存しながら`ne`と`Te`が一致しました。この段階検証を基礎に、次節の自己無撞着モデルで局所電離・壁損失・RF吸収を連成しています。

## 二ゾーン自己無撞着グローバルモデル

二つの局所粒子収支と二つの電子エネルギー収支を、非線形ngspice回路へ反復連成しました。基準条件は6反復で収束し、`ne,W=2.2889e14 m^-3`、`ne,F=2.2781e14 m^-3`、Bohmイオン束不均一度は`0.474%`でした。横RF枝はWaferからFocusへ`0.735 W`を移送しており、この条件では粒子・熱交換よりも電気的な横結合が局所電力均一化を支配します。

## 二ゾーン均一性最適化

Focus電源の振幅・位相とFocus ESC容量を設計変数にし、凍結プラズマによる高速選別と自己無撞着モデルによる再検証を組み合わせました。選択値は`99.88 V peak`、`+0.18°`、`184.05 pF`で、Bohmイオン束不均一度は`0.474%`から`0.049%`へ低下しました。全吸収電力の変化は`+0.262%`で全制約を満たしましたが、最適点は小さな設定差に敏感なため、実機適用には寄生容量の同定と公差を含むロバスト最適化が必要です。

## Qucs-S RLC一ゾーン連成

Qucs-S 26.1.0から出力したダミーロードなしの直列RLC netlistを正規化し、`w_feed`へ一ゾーンAr-CCPモデルを接続しました。自己無撞着計算は8反復で収束し、`Te=4.7493 eV`、`ne=1.8892e14 m^-3`、プラズマ吸収電力`0.4340 W`、電力収支残差`0.0294%`でした。入力インピーダンスは`155.68-j1064.84 Ω`で強い容量性を示すため、次段階ではプラズマ負荷を用いた整合素子の再設計が必要です。

## Qucs-S RLC整合条件探索

直列L1、負荷側並列C、コイル有限Qを段階探索し、自己無撞着な採用点を`L1=5.0 µH`、`C1=1.0 nF`、`Csh=12 pF`、`Q=30`としました。50 Ω電源抵抗を除いた外部負荷は`22.45-j6.18 Ω`、`|Gamma|=0.3883`で、基準の`0.9908`から60.8%低下しました。プラズマ吸収電力は`7.611 W`へ増加しましたが、R1とコイルESRの合計損失は`60.90 W`に達するため、次段階では実部品制約と効率を含む多目的最適化が必要です。

## 出典

- [Schmidt et al. (2018), DOI](https://doi.org/10.1088/1361-6595/aae429)
- [arXiv full text](https://arxiv.org/abs/1804.05638)
- [ngspice documentation](https://ngspice.sourceforge.io/docs.html)
