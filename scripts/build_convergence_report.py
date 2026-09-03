"""既存の検証JSONから、外部通信不要の収束技法レポートを生成する。"""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/data/schmidt2018_validation.json"
OUTPUT = ROOT / "reports/プラズマ等価回路_収束性改善ガイド.html"


def curve_plot(kind: str) -> str:
    """模式図ではなく、表示した無次元式を直接評価してSVGを生成する。"""
    electron = kind == "electron"
    xmin, xmax, ymax = (-1.0, 3.0, 3.0) if electron else (-3.0, 3.0, 5.0)
    xlabel = "Vs / Te" if electron else "Vs / δC"
    ylabel = "Ie / Isat" if electron else "C / [αC √(K/δC)]"
    pieces = [f'<svg viewBox="0 0 540 300" role="img" aria-label="{html.escape(ylabel)}の比較">',
              '<rect width="540" height="300" fill="white"/>']
    def point(x: float, y: float) -> str:
        return f"{62 + (x-xmin)/(xmax-xmin)*450:.2f},{248-y/ymax*205:.2f}"
    for y in range(int(ymax) + 1):
        py = 248-y/ymax*205
        pieces.append(f'<path d="M62 {py} H512" stroke="#dce5ec"/><text x="50" y="{py+4}" text-anchor="end">{y}</text>')
    for x in range(int(xmin), int(xmax)+1):
        px = 62+(x-xmin)/(xmax-xmin)*450
        pieces.append(f'<text x="{px}" y="268" text-anchor="middle">{x}</text>')
    for regularized, color in ((False, "#d97706"), (True, "#087e8b")):
        points = []
        for i in range(601):
            x = xmin+(xmax-xmin)*i/600
            if electron:
                # 図の平滑幅は見やすさのためδe/Te=0.2。実計算値とは区別する。
                y = math.exp(-(x+math.sqrt(x*x+0.2**2))/2) if regularized else math.exp(-x)
            else:
                if not regularized and x < 0.04:
                    continue
                y = (x*x+1)**(-0.25) if regularized else x**(-0.5)
            points.append(point(x, y))
        pieces.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>')
    pieces.append(f'<text x="64" y="22">{html.escape(ylabel)}</text><text x="288" y="292" text-anchor="middle">{xlabel}</text></svg>')
    return "".join(pieces)


def sensitivity_table(data: dict, name: str, field: str, label: str) -> str:
    rows = []
    for case in data[name]:
        m = case["metrics"]
        rows.append(f'<tr><td>{case[field]:g}</td><td>{m["absorbed_power_w"]:.6f}</td>'
                    f'<td>{m["plasma_voltage_amplitude_v"]:.6f}</td>'
                    f'<td>{m["input_impedance_real_ohm"]:.6f}</td>'
                    f'<td>{m["input_impedance_imag_ohm"]:.6f}</td></tr>')
    return f'<div class="scroll"><table><thead><tr><th>{label}</th><th>Ppl [W]</th><th>Vpl基本波 [V]</th><th>Re Zin [Ω]</th><th>Im Zin [Ω]</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def build() -> str:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    base = data["base_result"]["metrics"]
    power = [c["metrics"]["absorbed_power_w"] for c in data["regularization_sensitivity"]]
    span = 100*(max(power)-min(power))/base["absorbed_power_w"]
    source_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    template = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>プラズマ等価回路：正則化と収束性改善ガイド</title>
<style>
:root{--ink:#173046;--muted:#536779;--teal:#087e8b;--line:#dce5ec;--bg:#f2f6f9}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.85 "Yu Gothic","Meiryo",sans-serif}main{max-width:1120px;margin:auto;padding:32px 28px 70px}header{background:#123047;color:white;padding:44px;border-radius:18px}header p{color:#dce8ed}h1{font-size:32px;line-height:1.5;margin:10px 0}h2{font-size:25px;border-bottom:2px solid var(--line);padding-bottom:10px;margin:0 0 22px}h3{font-size:19px;margin:26px 0 8px}p{margin:10px 0 16px}section{background:white;border:1px solid var(--line);border-radius:14px;padding:30px;margin-top:22px;scroll-margin-top:20px}a{color:#07697a;overflow-wrap:anywhere}header a{color:white}.eyebrow{font-size:13px;letter-spacing:.14em}.lead{font-size:18px}.tags{display:flex;gap:8px;flex-wrap:wrap}.tag{border:1px solid #95bcc4;border-radius:20px;padding:2px 12px;font-size:13px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{padding:18px;background:#f2f7f9;border-radius:10px}.callout{border-left:5px solid var(--teal);padding:14px 18px;background:#edf8f8;margin:18px 0}.warn{border-color:#d97706;background:#fff8e9}nav{padding:20px 8px;display:flex;gap:10px 22px;flex-wrap:wrap}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:11px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}th{background:#eaf2f6}tbody tr:nth-child(even){background:#f9fbfc}.scroll{overflow-x:auto}code{font-family:Consolas,monospace;font-size:.93em;overflow-wrap:anywhere}pre{padding:18px;background:#102b40;color:#ecf6fa;border-radius:9px;white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 Consolas,monospace}.formula{background:#f0f5f8;padding:16px;border-radius:8px;font:17px/1.85 Consolas,"Meiryo",monospace;overflow-wrap:anywhere}.small,figcaption{font-size:13px;color:var(--muted)}figure{margin:0}svg{width:100%;height:auto}svg text{font:13px "Meiryo",sans-serif;fill:#30495e}.legend{font-size:13px}.orange{color:#a35b08}.teal{color:var(--teal)}li{margin:7px 0}.steps{counter-reset:step;list-style:none;padding:0}.steps li{padding:14px 16px;background:#f2f7f9;border-radius:8px}.steps li:before{counter-increment:step;content:counter(step) " / ";font-weight:bold;color:var(--teal)}footer{font-size:13px;padding:25px 5px;color:var(--muted)}
@media(max-width:720px){main{padding:12px}header,section{padding:22px 18px}.grid{grid-template-columns:1fr}h1{font-size:25px}h2{font-size:22px}table{min-width:650px}}
@media print{body{background:white;font-size:10pt}main{max-width:none;padding:0}header{background:white;color:var(--ink);padding:10px;border-radius:0}header p,header a{color:var(--ink)}nav{display:none}section{border:0;border-top:1px solid #aaa;border-radius:0;margin:10px 0;padding:18px 0}h2,h3{break-after:avoid}tr,figure,.card,.callout,pre{break-inside:avoid}.scroll{overflow:visible}pre{background:#f4f4f4;color:black}.grid{grid-template-columns:1fr 1fr}a{color:inherit} }
</style></head><body><main>
<header><div class="eyebrow">PLASMA × CIRCUIT / NUMERICAL PRACTICE</div><h1>正則化と収束性改善の実践ガイド</h1><p class="lead">外部RLC回路・非線形シース・グローバルモデルを、<br>「計算できる」から「信頼して比較できる」へ。</p><div class="tags"><span class="tag">ngspice / Qucs-S</span><span class="tag">1ゾーンを中心に解説</span><span class="tag">作成 2026-09-04</span></div><p class="small" style="color:#dce8ed">対象：本リポジトリの実装と既存検証値。外部通信・CDN不要、図表を内蔵した単体HTML。</p></header>
<nav aria-label="目次"><a href="#overview">1. 全体像</a><a href="#regularization">2. 正則化</a><a href="#transient">3. 過渡解析</a><a href="#coupling">4. 反復連成</a><a href="#evidence">5. 検証結果</a><a href="#diagnosis">6. 切り分け</a><a href="#recipe">7. 実行手順</a><a href="#sources">8. 根拠と限界</a></nav>
<section id="overview"><h2>1 / 収束には、異なる三つの階層がある</h2>
<div class="callout"><strong>結論：</strong>正則化だけで解決しない。モデルの有限性、過渡計算、RF周期性、外側の収支反復を順に確認し、最後に保存則と設定感度で解を検証する。</div>
<div class="scroll"><table><thead><tr><th>階層</th><th>解いている問題</th><th>典型的な失敗</th><th>確認方法</th></tr></thead><tbody>
<tr><td>① 回路の非線形解法</td><td>各時刻のKCL・素子方程式</td><td>指数発散、特異行列、timestep too small</td><td>ログ、有限値、接続・単位・符号</td></tr>
<tr><td>② RF周期定常</td><td>初期過渡が消えた繰り返し波形</td><td>計算は終わるがDCバイアスや包絡線が変動</td><td>隣接周期の電圧・電流L2差、平均電力</td></tr>
<tr><td>③ グローバルモデル固定点</td><td>吸収電力と粒子・エネルギー収支</td><td>密度が振動、遅い停滞、偽収束</td><td>更新量でなく未緩和の収支残差</td></tr></tbody></table></div>
<p>この一ゾーン定常Arモデルでは、電子温度は粒子収支から先に求め、密度を回路吸収電力と反復する。すべてのグローバルモデルで温度を固定できるわけではない。</p>
<p class="small">数値収束は、モデルの物理的妥当性や実験との一致を保証しない。また、消滅・多安定・周期倍化などを単なる数値不具合と決めつけない。</p></section>
<section id="regularization"><h2>2 / ゼロ近傍を滑らかに扱う</h2>
<h3>2.1 電子電流：負電圧側の指数増大を止める</h3><p>シース電圧は <code>Vs = Vbulk − Velectrode</code>。元の <code>Ie = Isat exp(−Vs/Te)</code> を負のVsまで外挿すると、電流とヤコビアンの勾配が大きくなる。障壁を滑らかな正部分に置き換える。</p>
<div class="formula">sδ(V) = [V + √(V² + δe²)] / 2<br>Ie,reg = Isat exp[−sδ(Vs)/Te]<br>0 ≤ dsδ/dV ≤ 1、理論上 0 &lt; Ie,reg ≤ Isat</div>
<p>ここではTeの数値をeV、VsをVで表すため指数は無次元。δe &gt; 0ならゼロ付近でも微分可能であり、<code>|dIe/dVs| ≤ Isat/Te</code> となる。浮動小数点では極端な正電圧でアンダーフローし、電流が0になることはある。</p>
<div class="grid"><figure>@@ELECTRON@@<figcaption>橙：元の指数式。青緑：正則化。形状を見やすくするため図だけδe/Te=0.2。実標準点は0.05/4.7493 ≈ 0.0105。</figcaption></figure><div class="card"><h3>実装上の工夫：負側だけ有理化</h3><p>大きな負Vでは <code>V + √(V²+δ²)</code> が桁落ちする。同じ関数を次式で評価する。</p><div class="formula">V ≤ 0：<br>sδ(V) = δ² / {2[√(V²+δ²) − V]}</div><p>分岐は数学的に同一の式の評価切り替えで、ハードクリップとは異なる。より極端な数値域ではV²のオーバーフロー対策も別途必要。</p></div></div>
<pre>.func spos(x,d) {(x &gt; 0) ? 0.5*(x+sqrt(x*x+d*d)) : 0.5*d*d/(sqrt(x*x+d*d)-x)}
Belectron1 plasma bulk1 I='iesat1*exp(-spos(v(bulk1,plasma),veps)/te_ev)'</pre>
<p class="small">単純なmax(V,0)は連続でも導関数に折れ点が残る。滑らか化は解法を助けるが、負Vsでの飽和モデルを選ぶ物理的仮定でもある。反転シースを定量的に解いたことにはならない。</p>
<h3>2.2 シース容量：1/√Vsの特異性を有限化</h3>
<div class="formula">K = 2 e ne ε0 A²<br>Creg(Vs) = αC √{K / √(Vs² + δC²)}<br>Creg(0) = αC √(K/δC)</div>
<div class="grid"><figure>@@CAPACITANCE@@<figcaption>橙：Vs &gt; 0の非正則化式。青緑：滑らかな拡張。図の縦軸はCreg(0)で正規化。負Vsへの延長はモデル上の便宜。</figcaption></figure><div class="card"><h3>小さい幅が常に良いわけではない</h3><p>δCを小さくすると元の式に近づくが、ゼロ付近の最大容量がδC<sup>−1/2</sup>で増え、過渡解法が厳しくなり得る。大きくすると計算が容易になっても物理応答を変える。</p><p>実装の初期候補はδe=δC=0.05 V。普遍的な最適値ではなく、後述の幅感度で許容性を判断する。</p></div></div>
<pre>Csh1 plasma bulk1 C='cscale*sqrt(ksh1/sqrt(v(bulk1,plasma)*v(bulk1,plasma)+vcap*vcap))'</pre>
<div class="callout warn"><strong>αCは収束調整パラメータではない。</strong>行列シースのQ=√(KVs)を微分するとdQ/dVs=0.5√(K/Vs)。新しい一ゾーンQucsモデルはαC=0.5。一方、Schmidt標準点再現の0.5862は公開点に合わせた経験的係数。δe・δCの滑らか化と混同しない。<a href="Capacitance規約監査.md">容量規約監査</a></div>
<p>ngspiceのこの電圧依存C指定は本プロジェクトのランプ試験で <code>I=C(V)dV/dt</code> と確認済み。電荷Qを指定する形式ならCではなく積分したQを与える。有限δのCregに対して、非正則化のQ式をそのまま使うと別モデルになる。</p></section>
<section id="transient"><h2>3 / 初期化・時間刻み・回路構造</h2>
<h3>3.1 電源を徐々に立ち上げる</h3><div class="formula">Vrf(t) = V0 tanh(t/τ) sin(2πft + φ)<br>τ = Nramp / f</div><p>Qucs一ゾーン実装はNramp=10。τで振幅は約76.2%、3τで約99.5%であり、「10周期で完全な定格」ではない。保存区間はランプ終了後かつ過渡消滅後にする。Schmidt再現deckは位相90°のSIN電源で、ランプを使用していない。</p>
<pre>* Qucs一ゾーンの立上げ例：位相0度、10周期を時定数とする
Bsource src 0 V='100*tanh(time/(10/13.56e6))*sin(2*pi*13.56e6*time)'</pre>
<h3>3.2 .icとuicは、初期条件を管理するために使う</h3><p>標準deckはシースに約28.806 Vの初期差電圧を与え、<code>uic</code>で初期DC動作点計算を省略する。無矛盾な定常状態を自動生成する指定ではない。初期値を変えて同じ周期解へ到達するかを別途調べる。</p><p>DC動作点が必要な回路にはgmin stepping・source steppingなどの補助がある。これらはDC解探索の手法であり、RFランプや周期収束判定の代替ではない。<a href="https://ngspice.sourceforge.io/docs/ngspice-manual.pdf">ngspice公式マニュアル（OP・IC・TRAN）</a></p>
<h3>3.3 Gear2・許容誤差・最大刻みを一組で管理</h3>
<pre>.options method=gear maxord=2 reltol=1e-6 abstol=1e-9 vntol=1e-7
* tran tstep tstop tstart tmax uic
tran 3.0727630285152412e-10 2.2123893805309735e-5 2.0353982300884957e-5 3.0727630285152412e-10 uic</pre>
<p>これは13.56 MHz・300周期・末尾24周期・最大刻みT/240の標準deck設定。tstart以前も計算する。240は均等な保存点数の保証ではなく、実際の保存点は適応刻みで増える。電子電流パルスはRF周期より短いため、基本波だけを見て刻みを決めない。</p>
<p>Gear2は数値減衰を持つため振動を抑えやすい一方、高調波や共振応答への影響を検証する必要がある。台形法との比較は今後の検証候補で、現時点で優劣の比較実験はしていない。reltolは相対、abstolは電流[A]、vntolは電圧[V]の許容値。収束を通すために緩めたら、最終値を厳しく戻して比較する。</p>
<h3>3.4 接続不備を数値設定で隠さない</h3><ul><li>DCで浮くノード、理想電圧源・インダクタだけのループ、電流源・容量だけで拘束された枝、接地・単位・素子名重複を確認する。</li><li>実部品にあるESR・有限Qは物理モデルとして入れる。例：指定周波数でRs=ωL/Q。ただし追加損失でプラズマの吸収電力も変わる。</li><li>診断用の並列抵抗やrshuntは漏れ電流と自己バイアスを変え得る。恒久採用するならV²/Rを収支へ含め、抵抗値感度を検証する。</li></ul></section>
<section id="coupling"><h2>4 / グローバルモデル反復を安定化する</h2>
<p>回路計算から得た密度目標をF(n)とする。高Qの整合近傍では、密度→負荷→整合→電力→密度のフィードバックが強くなり、単純置換 n←F(n) が振動し得る。</p>
<div class="formula">x = ln(n / nref)、r(x) = ln[F(n)/n]<br>xnext = x + λ r(x)　（0 &lt; λ ≤ 1）<br>nnext = n · [F(n)/n]<sup>λ</sup></div>
<p>対数空間の緩和は正の密度を保ち、桁の大きい変化を相対的に扱う。nrefは任意の固定基準密度で差分から消える。実装ではSI数値のlog差を使っている。λを小さくすると更新は穏やかになるが、収束保証ではない。</p>
<div class="callout warn"><strong>偽収束を防ぐ：</strong>更新率 |nnext−n|/n だけでは、λが小さいと誤って収束と判定する。現行実装は未緩和残差 <code>εn=|F(n)/n−1|=|expm1(r)|</code> を判定する。</div>
<h3>符号反転を見つけたら、対数密度の挟み撃ちへ</h3><p>現行実装はrの正負を記録し、両符号の点が得られたら対数座標の中点を次の候補にする。これは回路応答が同じ解の枝で連続的に評価される場合に有効。RF未収束、複数解、非単調な残差では、単純な符号追跡を厳密な大域収束保証と解釈しない。</p>
<h3>三つの受け入れ条件</h3><div class="scroll"><table><thead><tr><th>指標</th><th>定義と判定</th><th>扱い</th></tr></thead><tbody><tr><td>RF周期L2</td><td>εRF=‖y(t)−y(t−T)‖₂ / max(‖y(t)‖₂,小量)。電圧・電流とも&lt;2e−4</td><td>実装の収束判定。直近2周期を共通位相513点で補間比較</td></tr><tr><td>密度収支</td><td>εn&lt;1e−3、Schmidtは連続3回、Qucs一ゾーンは連続2回</td><td>実装の収束判定</td></tr><tr><td>電力収支</td><td>εP=|Psrc−PRrf−PRm−PRstray−Ppl|/|Psrc|。標準点の目安&lt;1e−3</td><td>独立の検証ゲート。現行密度反復の自動停止条件そのものではない</td></tr></tbody></table></div>
<p>上式の電力収支はSchmidt回路のポート分割。回路を変更したら追加損失を反映する。過渡中は蓄積エネルギー変化もあるので、定常周期で評価する。電力が極小の条件には絶対残差も必要。</p>
<p>整合容量の外反復も正値を保つ対数緩和を使用する（標準λ=0.7）。固定プラズマ整合→密度更新→再整合の内外反復は、内側のRF解が十分収束していることが前提。</p></section>
<section id="evidence"><h2>5 / 既存計算で確認できたこと</h2><p><strong>既存データの再集計であり、このレポート作成時に感度スイープを再実行した結果ではない。</strong>密度1.2451600902×10¹⁵ m⁻³、Te=4.7492937 eVを固定。αC=0.5862のSchmidt標準点であり、新しいQucs-RLC回路への直接保証ではない。</p>
<h3>5.1 正則化幅を100倍変えた比較</h3><p>δe=δCを同時に変更した試験。電子電流側と容量側の影響は、この試験だけでは分離できない。</p>@@REGTABLE@@
<p>Pplの全幅（最大−最小）/基準値は <strong>@@SPAN@@%</strong>。微小であるが、正則化なしより速い、失敗率が低い、という比較の証拠ではない。</p>
<h3>5.2 最大刻みT/Nの比較</h3>@@STEPTABLE@@<p>Im(Zin)は0.285～0.998 Ωへ変動しており、単調な刻み収束ではない。整合近傍の虚部を百分率だけで評価しない。Pplや基本波振幅が安定でも、高精度な整合条件を主張するにはさらに厳しい許容誤差・長い過渡時間・高調波の確認が必要。</p>
<div class="grid"><div class="card"><strong>標準点の電力収支残差</strong><div class="formula">@@POWERERROR@@ %</div><p>電源・外部損失・プラズマ吸収の閉じ方を確認。</p></div><div class="card"><strong>隣接RF周期差</strong><div class="formula">電圧 @@VERROR@@<br>電流 @@IERROR@@</div><p>周期性と収支の両方で確認する。</p></div></div>
<h3>未検証の比較は、未検証として扱う</h3><ul><li>正則化あり／なしの失敗率・実行時間・Newton反復回数の比較。</li><li>Gear2対台形法、ランプあり／なし、初期値複数条件からの同一解到達。</li><li>δeとδCの独立スイープ、および各条件で密度も再収束させる完全連成感度。</li><li>広い圧力・電源条件、Wafer―Focus二ゾーンでの同じ数値設定の有効性。</li></ul></section>
<section id="diagnosis"><h2>6 / 症状から原因を切り分ける</h2><div class="scroll"><table><thead><tr><th>症状</th><th>最初に確認すること</th><th>順番に試す対策</th><th>通った後の検証</th></tr></thead><tbody>
<tr><td>開始直後に特異行列</td><td>接地、DC経路、理想源の拘束</td><td>回路をダミー負荷へ戻す→接続修正→必要な物理損失</td><td>解析RLCと一致、診断用漏れ抵抗の影響</td></tr>
<tr><td>Vs≈0で刻みが縮む</td><td>Vs符号、exp・sqrtの定義域</td><td>sposと容量正則化→ランプ→刻み・許容誤差の確認</td><td>幅を小さくして主要量が維持されるか</td></tr>
<tr><td>ngspice終了でも波形が動く</td><td>DCバイアス・包絡線・εRF</td><td>計算周期数を増す、保存区間を後ろへ</td><td>複数周期窓で同じ平均・高調波</td></tr>
<tr><td>密度が上下に振れる</td><td>内側RF収束、残差符号</td><td>対数緩和→同じ解の枝で符号区間を使う</td><td>未緩和εn、複数初期密度</td></tr>
<tr><td>更新量だけが小さい</td><td>λが小さすぎないか</td><td>εnで判定し直す。反復上限を成功扱いしない</td><td>RFと収支の同時合格</td></tr>
<tr><td>吸収電力が負／損失と合わない</td><td>電流計の方向、端子、積分区間</td><td>0V計測源の正端子→負端子を正に統一</td><td>KCL、定常周期の電力収支、蓄積変化</td></tr>
<tr><td>収束したが整合点がずれる</td><td>αC、ESR、周波数、単位、刻み</td><td>数値感度と物理パラメータ感度を別々に実施</td><td>基本波複素Zと高調波・電力を併記</td></tr></tbody></table></div></section>
<section id="recipe"><h2>7 / 再現可能な実行・評価手順</h2>
<div class="scroll"><table><thead><tr><th>設定</th><th>Schmidt標準再現</th><th>Qucs一ゾーンRLC</th></tr></thead><tbody><tr><td>δe / δC [V]</td><td>0.05 / 0.05</td><td>0.05 / 0.05</td></tr><tr><td>αC</td><td>0.5862（公開点校正）</td><td>0.5（物理的微分容量）</td></tr><tr><td>電源ランプ</td><td>なし・cos位相</td><td>tanh・時定数10周期・sin位相</td></tr><tr><td>全周期 / 保存周期 / T刻み分割</td><td>300 / 24 / 240</td><td>960 / 32 / 400</td></tr><tr><td>密度緩和λ / 連続合格回数</td><td>0.45 / 3</td><td>0.4 / 2</td></tr></tbody></table></div>
<p class="small">設定元：configs/schmidt2018.json、configs/qucs_rlc_one_zone.json。両者は回路も容量規約も違うため、960周期の方が優秀という比較ではない。</p>
<ol class="steps"><li>ダミーRLC負荷で電圧・電流・消費電力を解析値と比較する。</li><li>ne・Teを固定し、正則化した非線形プラズマを接続する。</li><li>RF周期性と電力収支が通ってから、外側の密度反復を有効にする。</li><li>最大刻み、全周期数、許容誤差、δe、δCを一項目ずつ変え、差を記録する。</li><li>最終候補は完全連成をやり直し、別初期値でも同じ解か確認する。</li><li>netlist・設定・ngspice版・ログ・残差履歴・波形・実行時間・失敗理由を保存する。</li></ol>
<h3>PowerShell：既存テストとレポートの再生成</h3><pre>Set-Location "C:\Users\TK\Documents\ChatGPT\プラズマ等価回路"
uv run pytest -q
uv run python scripts/build_convergence_report.py</pre>
<h3>感度計算を再実行する場合（数分以上かかる場合がある）</h3><p>既存の検証JSONを上書きしないよう、再実行結果は新しいartifactsフォルダへ保存する。</p><pre>uv run plasma-reproduce --config configs/schmidt2018.json --output artifacts/convergence_review/base
uv run plasma-validate --config configs/schmidt2018.json `
  --base-summary artifacts/convergence_review/base/summary.json `
  --base-output artifacts/convergence_review/base `
  --raw-output artifacts/convergence_review/sweeps `
  --summary-output artifacts/convergence_review/validation.json `
  --figure-directory artifacts/convergence_review/figures</pre>
<p>上記感度CLIは固定ne・Teの試験。再生成スクリプトは追跡済みのreports/data JSONを読むため、この再実行結果へ自動的に切り替わらない。比較・採否を確認してからデータの更新を行う。</p></section>
<section id="sources"><h2>8 / 根拠・実装への対応・限界</h2><p>本文の数式は実装から整理した説明。数値は既存の保存結果を再集計し、今回新しい物理検証を行ったとは主張しない。</p>
<ul><li><a href="data/schmidt2018_validation.json">検証JSON</a>：基準値、時間刻み・幅・容量規約の感度。</li><li><a href="../src/plasma_circuit/ngspice.py">ngspice.py</a>：spos、C指定、初期値、Gear2、波形解析。</li><li><a href="../src/plasma_circuit/coupling.py">coupling.py</a>／<a href="../src/plasma_circuit/qucs_one_zone.py">qucs_one_zone.py</a>：対数緩和、符号追跡、未緩和残差・RF判定。</li><li><a href="../tests/test_physics.py">test_physics.py</a>／<a href="../tests/test_ngspice_capacitor.py">test_ngspice_capacitor.py</a>：有限性、飽和上限、微分容量規約。</li><li><a href="Capacitance規約監査.md">容量規約監査</a>：物理的0.5と校正0.5862の区別。</li><li><a href="https://ngspice.sourceforge.io/docs/ngspice-manual.pdf">ngspice公式マニュアル</a>：OP・TRAN・IC・OPTIONS・behavioral source。参照確認日2026-09-04。</li><li><a href="https://arxiv.org/abs/1804.05638">Schmidt et al. (2018)</a>：対象論文。ここに示した全ての正則化技法が論文由来という意味ではない。</li></ul>
<div class="callout">次の検証としては、<strong>δe×δCの独立感度 → ランプ／初期値比較 → 完全連成再収束</strong>を推奨する。成功率と計算時間も同時に記録すれば、物理量への影響と計算上の利益を分けて評価できる。</div>
<p class="small">元JSON SHA-256：<code>@@HASH@@</code><br>HTML内の図は無次元式から生成。実測・シミュレーション波形ではない。データ表は上記JSONから直接生成。生成コード：scripts/build_convergence_report.py。</p></section>
<footer>このHTMLは単体で閲覧・印刷できます。相対リンク先のコード・JSONはリポジトリと一緒に配置した場合に利用できます。</footer>
</main></body></html>'''
    replacements = {
        "@@ELECTRON@@": curve_plot("electron"),
        "@@CAPACITANCE@@": curve_plot("capacitance"),
        "@@REGTABLE@@": sensitivity_table(data, "regularization_sensitivity", "regularization_width_v", "δe=δC [V]"),
        "@@STEPTABLE@@": sensitivity_table(data, "timestep_sensitivity", "samples_per_cycle", "分割N（tmax=T/N）"),
        "@@SPAN@@": f"{span:.5f}",
        "@@POWERERROR@@": f'{100*base["power_balance_relative_error"]:.5f}',
        "@@VERROR@@": f'{base["cycle_l2_voltage"]:.3e}',
        "@@IERROR@@": f'{base["cycle_l2_current"]:.3e}',
        "@@HASH@@": source_hash,
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    if "@@" in template:
        raise ValueError("未置換のテンプレート項目があります")
    return template


if __name__ == "__main__":
    OUTPUT.write_text(build(), encoding="utf-8", newline="\n")
    print(OUTPUT)
