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
```

生成途中の netlist、波形、ログは `artifacts/` に保存されます。検証済みの集計値は `reports/data/`、図とレポートは `reports/` に保存します。結果の要約は [Schmidt2018 再現レポート](reports/Schmidt2018再現レポート.md) を参照してください。

## 数値安定化

論文のシース容量 `C_s∝1/sqrt(V_s)` は `V_s=0` で発散し、電子電流 `I_e∝exp(-V_s/T_e)` は負のシース電圧で非物理的に増大します。本実装では次を使用します。

```text
Vabs_delta(V) = sqrt(V^2 + delta_C^2)
Vplus_delta(V) = (V + sqrt(V^2 + delta_e^2)) / 2
C_s,reg = alpha_C sqrt(K / Vabs_delta(V_s))
I_e,reg = I_e,sat exp(-Vplus_delta(V_s) / T_e)
```

これにより容量は有限となり、電子電流は滑らかに `0 < I_e <= I_e,sat` を満たします。既定の `delta_C=delta_e=0.05 V` は 1 V 以上で元式との差が十分小さく、感度試験で確認します。

`alpha_C=0.5862` は、論文に netlist と非線形容量の SPICE 実装規約が掲載されていないため、公開された密度・整合容量で `Im(Z_in)` を合わせて同定した唯一の再現係数です。この一点以外を使わずに、論文の電圧、各枝電流、各損失電力を検証しています。係数を 1 とする論文式の直読ケースと、`Q=C(V)V` から得られる 0.5 のケースも感度結果へ保存しています。

## 標準点の結果

- `T_e = 4.7493 eV`、`n_e = 1.2452e15 m^-3`
- `Z_in = 49.55 + j0.40 ohm`
- プラズマ電圧振幅 `355.3 V`、DC オフセット `-244.5 V`
- `I_rf = 1.004 A`、`I_pl = 0.604 A`、`I_L = 6.659 A`、`I_stray = 6.055 A`
- 損失は電源抵抗 `25.22 W`、整合器 `11.08 W`、寄生枝 `9.18 W`、プラズマ `4.72 W`
- 全系電力収支残差 `0.021%`

## 出典

- [Schmidt et al. (2018), DOI](https://doi.org/10.1088/1361-6595/aae429)
- [arXiv full text](https://arxiv.org/abs/1804.05638)
- [ngspice documentation](https://ngspice.sourceforge.io/docs.html)
