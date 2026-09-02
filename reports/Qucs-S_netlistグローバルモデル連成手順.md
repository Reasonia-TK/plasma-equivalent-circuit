# Qucs-S netlistとグローバルモデルの連成手順

## 1. 目的

Qucs-Sで作成した外部回路をngspice形式のnetlistとして出力し、Pythonで実装したプラズマグローバルモデルと自己無撞着に反復連成する手順を定める。

対象はWafer電極とFocus-ring電極を持つ二ゾーンモデルとし、既存の次の状態量を回路計算と連成する。

```text
ne,W, ne,F, Te,W, Te,F
```

Qucs-Sは回路図入力と外部回路の管理に使用し、反復制御、プラズマパラメータ更新、ngspice実行、波形解析、粒子・エネルギー収支はPythonが担当する。

## 2. 推奨方式

最初の実装では、Qucs-Sに回路全体を持たせるのではなく、責務を次のように分離する。

| 領域 | 管理するソフトウェア | 内容 |
|---|---|---|
| 外部回路 | Qucs-S | RF電源、整合回路、ケーブル、ESC容量、誘電損失、寄生素子 |
| プラズマ等価回路 | Python管理のSPICE include | 非線形シース、電子・イオン電流、bulk R–L、横方向R–L結合 |
| グローバルモデル | Python | 粒子収支、電子エネルギー収支、`ne`・`Te`更新 |
| 過渡解析 | ngspice | RF波形、平均吸収電力、シース電圧、入力インピーダンス |

この構成では外部回路をQucs-S上で自由に変更でき、プラズマモデルは既存の検証済み実装を再利用できる。Qucs-Sはngspiceを推奨バックエンドとしており、`.PARAM`およびパラメータ化したSPICE回路を利用できる。

### 2.1 実装済みの一ゾーン最小構成

本手順の最初の段階として、Wafer側だけの直列RLC回路を実装済みである。Qucs-S側の必須インターフェースはWire Label `w_feed`で、電流計測用0 V電圧源はPythonが正規化後に挿入する。このため、一ゾーン最小構成ではQucs-S回路に電流計を置かなくてもよい。

```powershell
uv run plasma-qucs-one-zone --config configs/qucs_rlc_one_zone.json --output artifacts/qucs_rlc_one_zone/self_consistent
```

実装内容、数値条件、収束結果は[Qucs-S RLC一ゾーン連成レポート](Qucs-S_RLC一ゾーンプラズマ連成.md)を参照する。以下はこの一ゾーン実装をWafer―Focus ring二ゾーンへ拡張するための設計手順である。

## 3. 全体フロー

```text
Qucs-S schematic (.sch)
        |
        | ngspice形式へ出力
        v
external_circuit.net
        |
        v
Python netlist assembler
  +-- runtime_params.inc   <- 現在のne, Teから毎反復生成
  +-- external_circuit.inc <- Qucs-S出力を正規化
  +-- plasma_two_zone.inc  <- 非線形プラズマ等価回路
  +-- analysis.control     <- tran、wrdata、収束設定
        |
        v
case.cir
        |
        v
C:\Spice64\bin\ngspice_con.exe
        |
        v
電圧・電流波形
        |
        v
局所吸収電力、シース電圧、回路収支
        |
        v
二ゾーングローバルモデル
        |
        v
新しいne,W、ne,F、Te,W、Te,F
        |
        +---------- 収束まで反復 ----------+
```

## 4. 前提条件

- Windows 10/11
- Qucs-S
- Qucs-Sのシミュレーションバックエンドをngspiceへ設定
- ngspice：`C:\Spice64\bin\ngspice_con.exe`
- このリポジトリのPython・`uv`環境
- Qucs-S回路から参照する`.lib`、`.model`、`.include`ファイル

Qucs-Sの`Simulation > Simulators Settings`で、可能なら連成計算と同じngspice実行ファイルを設定する。Qucs-S付属版と連成計算用ngspiceのバージョンが異なると、非線形素子や収束条件の差が比較へ混入する可能性がある。

## 5. Qucs-Sで作成する外部回路

### 5.1 対象要素

Qucs-S側には次を配置する。

- Wafer RF電源
- Focus-ring RF電源
- 電源内部抵抗
- 並列・直列整合素子
- ケーブルまたは伝送線路の等価回路
- ESC結合容量
- 誘電体ESR、漏れ抵抗
- 寄生容量・寄生インダクタンス
- 電流測定用0 V電圧源

### 5.2 固定する接続ノード名

Python側のプラズマ回路へ接続するノード名を回路インターフェースとして固定する。

```text
wafer    : Wafer表面ノード
focus    : Focus-ring表面ノード
w_zone   : Wafer側局所bulkノード
f_zone   : Focus側局所bulkノード
0        : 接地
```

Qucs-SのWire Labelを使用して明示的に命名する。自動生成ノード番号へ依存しない。

### 5.3 固定する計測素子名

電力計算に必要な枝へ0 V電圧源を挿入し、参照名を固定する。

一ゾーン最小構成では、Pythonが`w_feed`の直後に`Vsense_surface_wafer`、電源の直後に`Vsense_generator_qucs`を自動挿入するため、Qucs-S側の計測素子は不要である。二入力・二ゾーンへ拡張してQucs-S側に複数の分岐を持たせる場合は、次のRefDesを固定する方式が有効である。

```text
VMEAS_W_GENERATOR
VMEAS_F_GENERATOR
VMEAS_W_DIELECTRIC
VMEAS_F_DIELECTRIC
VMEAS_W_SURFACE
VMEAS_F_SURFACE
```

横方向結合電流はプラズマinclude内の`L_LATERAL`で計測する。Qucs-Sのバージョンや素子形式によってRefDesが変換される可能性があるため、Pythonは実行前に必要なノード名と素子名を検査する。

## 6. 動的プラズマパラメータ

反復ごとに変化する値は固定数値ではなく、SPICEパラメータとして参照する。

| パラメータ | 意味 |
|---|---|
| `tew`, `tef` | Wafer/Focus電子温度 |
| `kshw`, `kshf` | Powered sheath容量係数 |
| `kshgw`, `kshgf` | Ground sheath容量係数 |
| `iesatw`, `iesatf` | Powered sheath電子飽和電流 |
| `iesatgw`, `iesatgf` | Ground sheath電子飽和電流 |
| `iionw`, `iionf` | Powered sheathイオン電流 |
| `iiongw`, `iiongf` | Ground sheathイオン電流 |
| `rbulkw`, `rbulkf` | 局所bulk抵抗 |
| `lbulkw`, `lbulkf` | 局所bulkインダクタンス |
| `rlat`, `llat` | 横方向結合の抵抗・インダクタンス |
| `veps`, `vcap` | 電子電流・容量の滑らか化電圧 |
| `cscale` | シース微分容量係数 |

Qucs-SまたはSPICE includeの素子値では、パラメータを波括弧で参照する。

```spice
Rbulk_w w_bulk_mid w_zone {rbulkw}
Lbulk_w w_sheath_bulk w_bulk_mid {lbulkw}
Iion_w w_sheath_bulk wafer DC {iionw}

R_LATERAL w_zone lateral_mid {rlat}
L_LATERAL lateral_mid f_zone {llat}
```

Qucs-Sのngspice用`.PARAM`は解析前に評価される。結果処理用のEquationは解析後に評価されるため、回路素子値の更新にはEquationではなく`.PARAM`を使う。

## 7. Qucs-S netlistの出力と管理

### 7.1 出力ファイル

Qucs-Sが生成するngspice netlistを保存する。バージョンによって`.net`または`.cir`になる可能性があるが、拡張子ではなくngspice互換の内容を使用する。

元の出力ファイルは生成物として保存し、Pythonから直接上書きしない。

```text
qucs/
  esc_two_zone_prj/
    esc_two_zone.sch
  exported/
    esc_two_zone_raw.net
```

### 7.2 外部ファイル

Qucs-S回路が参照するファイルもリポジトリ内の専用ディレクトリへ置く。

```text
qucs/models/
  cable_model.lib
  matching_network.lib
```

絶対パスへの依存を避け、Pythonが反復ディレクトリへコピーまたは解決できる相対パスにする。

## 8. netlist正規化

Qucs-S出力をSPICE includeとして再利用する前に、Pythonの正規化処理で次を行う。

1. 文字コードと改行を正規化する。
2. Qucs-Sが出力した`.tran`、`.ac`、`.dc`を除去する。
3. 既存の`.control`から`.endc`を除去する。
4. 末尾の`.end`を除去する。
5. `.include`と`.lib`のパスを検査する。
6. 必須ノード・計測素子の存在を検査する。
7. 禁止された解析命令や未対応バックエンド構文がないことを検査する。
8. 正規化後の内容を`external_circuit.inc`として反復ディレクトリへ保存する。

解析条件をPython側へ一元化することで、Qucs-S GUI実行とグローバルモデル反復で異なる`.tran`設定が混在することを防ぐ。

## 9. 反復パラメータファイル

Pythonは現在の`ne`、`Te`から回路パラメータを計算し、各反復で`runtime_params.inc`を生成する。

```spice
.param tew=3.9774323291
.param tef=3.9774274668
.param veps=5.0e-2
.param vcap=5.0e-2
.param cscale=5.0e-1
.param kshw=...
.param kshf=...
.param kshgw=...
.param kshgf=...
.param iesatw=...
.param iesatf=...
.param iesatgw=...
.param iesatgf=...
.param iionw=...
.param iionf=...
.param iiongw=...
.param iiongf=...
.param rbulkw=...
.param lbulkw=...
.param rbulkf=...
.param lbulkf=...
.param rlat=...
.param llat=...
```

各反復のファイルを残すことで、使用したプラズマ状態とnetlistを後から追跡できる。

## 10. 最終netlistの組み立て

Pythonが各反復用の`case.cir`を次の構成で生成する。

```spice
Qucs-S external circuit coupled to two-zone global plasma model

.include "runtime_params.inc"
.include "external_circuit.inc"
.include "plasma_two_zone.inc"

.ic v(wafer)=0 v(focus)=0 v(w_zone)=40 v(f_zone)=40
.options method=gear maxord=2 reltol=1e-5 abstol=1e-10 vntol=1e-7

.control
set noaskquit
set wr_singlescale
set wr_vecnames
option numdgt=15
tran 4.1e-10 35.4u 33.6u 4.1e-10 uic
wrdata waveforms.dat \
  v(wafer) v(focus) v(w_zone) v(f_zone) v(w_zone,f_zone) \
  i(VMEAS_W_GENERATOR) i(VMEAS_F_GENERATOR) \
  i(VMEAS_W_SURFACE) i(VMEAS_F_SURFACE) i(L_LATERAL)
quit
.endc
.end
```

実際の時間刻み、解析時間、保存開始時刻は既存の`configs/esc_two_zone.json`から計算する。

## 11. ngspiceの実行

現在の実装と同じ実行ファイルを使用する。

```powershell
& "C:\Spice64\bin\ngspice_con.exe" -n -o ngspice.log case.cir
```

各反復では次を保存する。

```text
iteration_00/
  case.cir
  runtime_params.inc
  external_circuit.inc
  plasma_two_zone.inc
  waveforms.dat
  ngspice.log
  process.json
```

`process.json`にはコマンド、終了コード、入力した`ne`・`Te`、計算した回路パラメータを記録する。

## 12. 波形から計算する量

保存波形から既存実装と同じ方法で次を計算する。

- Wafer/Focusポート吸収電力
- Wafer/Focus局所配分電力
- WaferからFocusへの横RF電力移送
- Powered/ground sheath平均電圧
- Wafer/Focus入力インピーダンス
- 電源供給電力
- 外部回路損失
- 全回路電力収支残差
- 最終周期間L2差

符号規約を固定し、0 V計測源の正端子から負端子へ流れる電流を正とする。Qucs-S GUIの電力表示へ依存せず、Pythonが同一の電圧・電流積分式で計算する。

## 13. グローバルモデル反復

既存の二ゾーン反復を、回路runnerだけ差し替えて使用する。

```python
state = initial_state

for iteration in range(max_iterations):
    plasma = compute_two_zone_parameters(config, state)
    write_runtime_parameters(plasma)

    simulation = run_qucs_two_zone_ngspice(
        config=config,
        plasma=plasma,
        case_directory=iteration_directory,
    )

    powers = allocate_local_powers(simulation.metrics)
    sheaths = extract_mean_sheath_voltages(simulation.metrics)
    target = solve_two_zone_balance_state(config, powers, sheaths, state)
    next_state = relax_states(state, target.states)

    if fixed_point_converged(state, target, simulation):
        break

    state = next_state
```

既存の`run_two_zone_ngspice()`を置き換えず、次の二つを並存させる。

```text
run_generated_two_zone_ngspice()  : 現行Python生成回路
run_qucs_two_zone_ngspice()       : Qucs-S出力回路
```

波形解析とグローバルモデルは共通化し、同じ物理条件で両回路を比較できる構造にする。

## 14. 段階的な検証

### 段階1：netlist単体検証

1. Qucs-S GUIで固定プラズマ条件を実行する。
2. 同じ出力netlistを`ngspice_con.exe`から直接実行する。
3. 代表ノード電圧、枝電流、周期平均電力を比較する。

### 段階2：既存回路との比較

1. 現行Python生成回路と同じ部品値をQucs-Sへ設定する。
2. `ne`と`Te`を固定する。
3. 吸収電力、シース電圧、入力インピーダンス、横RF電力を比較する。

### 段階3：パラメータ更新検証

1. `ne`を基準から`-10%`、`+10%`変更する。
2. bulk抵抗・インダクタンス、シース係数、電流源が期待方向へ変化することを確認する。
3. `Te`についても同様に確認する。

### 段階4：一反復検証

固定状態の回路結果から局所電力・シース電圧を計算し、グローバルモデルが有限で正の目標状態を返すことを確認する。

### 段階5：完全連成

固定点まで反復し、次の合格条件を確認する。

| 検証項目 | 合格条件 |
|---|---:|
| RF周期L2差 | `< 2e-4` |
| 回路電力収支残差 | `< 1e-3` |
| グローバルモデル最大残差 | `< 2e-4` |
| 状態量 | `ne > 0`、`Te > 0`、有限 |
| Qucs/Python基準点差 | 主要指標で `< 0.5%` |

`0.5%`は初期導入時の工学的ゲートであり、差の原因を整理した後に縮小する。

## 15. 実装ファイル案

```text
qucs/
  esc_two_zone_prj/
  exported/
  models/

src/plasma_circuit/
  qucs_netlist.py          # 読み込み、正規化、契約検査
  qucs_two_zone_runner.py  # include生成、ngspice実行

configs/
  qucs_two_zone.json
  qucs_two_zone_contract.json

tests/
  test_qucs_netlist.py
  test_qucs_two_zone_runner.py

reports/
  data/qucs_two_zone_baseline.json
```

契約ファイルには必要なノード、素子、出力ベクトルを定義する。

```json
{
  "required_nodes": ["wafer", "focus", "w_zone", "f_zone"],
  "required_devices": [
    "VMEAS_W_GENERATOR",
    "VMEAS_F_GENERATOR",
    "VMEAS_W_SURFACE",
    "VMEAS_F_SURFACE"
  ],
  "forbidden_directives": [".tran", ".ac", ".dc", ".control", ".end"]
}
```

## 16. 想定される注意点

1. QucsatorRF形式のnetlistはngspice互換ではないため、必ずngspiceバックエンドを選ぶ。
2. Qucs-SのEquationは解析後処理用であり、反復パラメータには`.PARAM`を使用する。
3. Qucs-Sが生成する解析命令とPython生成の`.control`を重複させない。
4. `.end`は最終`case.cir`に一つだけ置く。
5. 相対`.include`は反復ディレクトリを基準に解決できるようにする。
6. RefDesの変換に備え、netlist契約検査を必須にする。
7. Qucs-Sと直接実行で同じngspiceバージョンを使用する。
8. GUI表示値ではなく、保存波形をPythonで再積分して比較する。
9. Qucs-S netlist更新時は、正規化後ファイルの差分と基準点回帰試験を実行する。

## 17. 導入順序

最小構成は次の順で進める。

1. Qucs-SでWafer側だけの線形RLC回路を作成する。
2. netlist正規化と直接ngspice実行を実装する。
3. Qucs-S GUIとの波形一致を確認する。
4. 二入力外部回路へ拡張する。
5. 固定プラズマincludeを接続する。
6. `runtime_params.inc`生成を追加する。
7. 一反復のグローバルモデルへ接続する。
8. 完全な二ゾーン固定点反復へ進む。
9. 現行Python生成回路との比較レポートを作成する。

この順序なら、Qucs-S変換、ngspice実行、プラズマモデル、固定点反復を一度に変更せず、問題を層ごとに切り分けられる。

## 18. 実装開始時に必要な入力

- Qucs-Sの`.sch`ファイル
- Qucs-Sが生成した`.net`または`.cir`
- 使用しているQucs-Sバージョン
- Qucs-Sで選択したシミュレーションバックエンド
- 参照する`.lib`、`.model`、`.include`ファイル
- Wafer/Focus/groundへ対応するノード名
- 測定したい電圧・電流・電力

これらをリポジトリへ配置した後、まず固定プラズマ状態の比較から実装を開始する。

## 19. 参考資料

- [Qucs-S公式サイト](https://ra3xdh.github.io/)
- [Qucs-S公式ドキュメント](https://app.readthedocs.org/projects/qucs-s-help/downloads/pdf/latest/)
- [Qucs-S GitHubリポジトリ](https://github.com/ra3xdh/qucs_s)
- [ngspice公式ドキュメント](https://ngspice.sourceforge.io/docs.html)
- [ngspice User's Manual](https://ngspice.sourceforge.io/docs/ngspice-manual.pdf)
- [二ゾーン自己無撞着モデル](ESC_二ゾーン自己無撞着モデル.md)
- [二ゾーン均一性最適化](ESC_二ゾーン均一性最適化.md)
