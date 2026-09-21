# FedPACT 報告会用フロー図

報告会用の粒度に絞った Mermaid 原稿。各図は PowerPoint 1 枚を想定する。

- 反映基準: 2026年9月16日 09:00 のZettsu教授による処理フロー修正指示
- 現在の実装方針: Phase I Basic path、FedAvg、client側classification headも更新・集約・再配布
- 9月1日PDFの層名は「代理データとlogitsのペア」だが、現行Stage Iでは`q`・`μ`をsoftmax probabilityに固定したため、図中は「出力確率のペア」と表記する
- 最新実装ログの数値結果は動作確認段階であり、手法フローの処理としては追加しない

## 1. FedPACT の全体構造

説明の中心: privateデータを送らず、clientで得た局所適応挙動を三層で基盤モデルへ移す。第1層と第3層の更新状態は、それぞれ次ラウンドへ戻る。

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"linear","nodeSpacing":38,"rankSpacing":52},"themeVariables":{"fontFamily":"Yu Gothic, Meiryo, sans-serif","fontSize":"20px","lineColor":"#455A64"}}}%%
flowchart LR
    subgraph LAYER1["第1層：surrogateモデルの連合継続学習"]
        direction TB
        C["クライアント<br/>局所適応と<br/>Q<sub>k,r</sub>生成"]
        AGG["Phase I Basic path<br/>LoRA・classification headを<br/>FedAvgで集約"]
        MSTATE["global surrogate state<br/>M<sub>G</sub><sup>t</sup> → M<sub>G</sub><sup>t+1</sup>"]
        QOUT["client / prototype単位<br/>Q<sub>k,r</sub> = (μ<sub>k,r</sub><sup>post</sup>, Δμ<sub>k,r</sub>, π<sub>k,r</sub>)"]

        C -->|"LoRA・head更新"| AGG
        AGG --> MSTATE
        C -->|"prototype"| QOUT
    end

    subgraph LAYER2["第2層：代理データと出力確率のペアの生成，選択，調整"]
        direction TB
        SEED["クラウド側シードデータ<br/>D<sub>seed</sub>"]
        L2["シード検索・変換<br/>代理データ生成<br/>選択・重み付け<br/>Stage I：出力確率"]
        SEED --> L2
    end

    subgraph LAYER3["第3層：異種モデル間の知識蒸留"]
        direction TB
        BASE["round t の基盤モデル<br/>予測 p<sub>F</sub>"]
        L3["P<sub>k,r</sub>をteacherとして<br/>基盤LoRAを更新"]
        FSTATE["更新後の基盤LoRA<br/>クラウドで保持"]

        BASE --> L3 --> FSTATE
        FSTATE -->|"次round"| BASE
    end

    MSTATE -->|"M<sub>G</sub><sup>t</sup>・M<sub>G</sub><sup>t+1</sup>"| L2
    QOUT -->|"FedAvgせずに渡す"| L2
    L2 -->|"P<sub>k,r</sub> = (x̃<sub>k,r</sub>, μ<sub>k,r</sub><sup>post</sup>)"| L3
    MSTATE -->|"M<sub>G</sub><sup>t+1</sup>・更新後headを<br/>次roundへ再配布"| C
    FSTATE -.->|"Later Phase<br/>補助的蒸留でglobal surrogate側へ還元"| MSTATE

    LEGEND["凡例<br/>M<sub>G</sub><sup>t</sup> / M<sub>G</sub><sup>t+1</sup>：連合集約前 / 後のglobal surrogate<br/>Q<sub>k,r</sub>：client k のprototype　π<sub>k,r</sub>：frequency<br/>D<sub>seed</sub>：server-side seed　P<sub>k,r</sub>：第3層へ渡す代理ペア　p<sub>F</sub>：foundationの予測<br/>実線：Basic path　破線・橙：Optional / Later Phase"]

    classDef client fill:#EAF2FF,stroke:#2563A6,stroke-width:2px,color:#102A43;
    classDef layer fill:#FFFFFF,stroke:#37474F,stroke-width:2px,color:#172B4D;
    classDef data fill:#F3E8FF,stroke:#7E57A8,stroke-width:2px,color:#2D1B4E;
    classDef state fill:#EAF7EA,stroke:#438A43,stroke-width:2px,color:#173A17;
    classDef optional fill:#FFF4DC,stroke:#C68A00,stroke-width:2px,stroke-dasharray:7 5,color:#5A3D00;
    classDef legend fill:#F7F9FB,stroke:#90A4AE,stroke-width:1px,color:#263238;
    classDef legendItem fill:transparent,stroke:transparent,color:#263238;

    class C client;
    class AGG,L2,L3 layer;
    class QOUT,SEED data;
    class MSTATE,BASE,FSTATE state;
    class LEGEND legend;
```

発表時の説明:

- 第1層では、LoRA・classification headの更新と`Q_(k,r)`を別経路で送る。FedAvgするのはLoRA・headであり、`Q_(k,r)`ではない。
- `M_G^t=S+A_G^t`は連合集約前、`M_G^(t+1)=S+A_G^(t+1)`は連合集約後のglobal surrogateである。両方を同じラウンドの第2層で使う。
- `M_G^(t+1)`は次ラウンドのクライアントへ再配布する。更新後の基盤LoRAはクラウドで次ラウンドへ引き継ぐ。
- 更新後foundationからglobal surrogate側への補助的蒸留はLater Phaseであり、更新方法と実行周期は未確定である。

## 2. 第1層：surrogateモデルの連合継続学習

説明の中心: `q^(pre)`から`Δq`までの順序を固定し、LoRA・headの集約経路とprototype経路を分ける。

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"linear","nodeSpacing":34,"rankSpacing":48},"themeVariables":{"fontFamily":"Yu Gothic, Meiryo, sans-serif","fontSize":"22px","lineColor":"#455A64"}}}%%
flowchart TB
    subgraph LAYER1["第1層：surrogateモデルの連合継続学習（Phase I Basic path：FedAvg）"]
        direction TB
        subgraph L1MAIN[" "]
        direction LR
        subgraph CLIENT["クライアント k"]
            direction LR
            PRE["1. q<sub>k,i</sub><sup>pre</sup>計算<br/>M<sub>G</sub><sup>t</sup>・同じx<sub>k,i</sub>"]
            LOCAL["2. privateデータで局所更新<br/>surrogate LoRA・classification head"]
            POST["3. q<sub>k,i</sub><sup>post</sup>計算<br/>局所学習後・同じx<sub>k,i</sub>"]
            DELTA["4. 出力変化<br/>Δq<sub>k,i</sub> = q<sub>k,i</sub><sup>post</sup> - q<sub>k,i</sub><sup>pre</sup>"]
            PROTO["5. class-wise aggregation<br/>sample単位のq<sup>post</sup>・Δqから生成<br/>Q<sub>k,r</sub> = (μ<sub>k,r</sub><sup>post</sup>, Δμ<sub>k,r</sub>, π<sub>k,r</sub>)"]
            KEEP["client内に保持<br/>privateデータ・raw sample・<br/>独立したq<sup>pre</sup>"]

            PRE --> LOCAL --> POST --> DELTA --> PROTO
            PRE --- KEEP
        end

        subgraph SERVER["クラウド"]
            direction LR
            AGG["LoRA・classification head更新を<br/>FedAvgで集約"]
            MGNEXT["連合集約後のglobal surrogate<br/>M<sub>G</sub><sup>t+1</sup> = S + A<sub>G</sub><sup>t+1</sup><br/>更新後classification head<br/>同一round第2層と次round clientへ"]
            L2USE["同一roundの第2層へ<br/>M<sub>G</sub><sup>t</sup>・M<sub>G</sub><sup>t+1</sup>・Q<sub>k,r</sub>"]

            AGG --> MGNEXT --> L2USE
        end

        LOCAL -->|"A<sub>k</sub><sup>t+1</sup>またはLoRA差分<br/>classification head更新"| AGG
        PROTO -->|"Q<sub>k,r</sub>はFedAvgしない"| L2USE
        MGNEXT -->|"LoRA・headを次roundへ再配布"| PRE
        end

        LEGEND["凡例<br/>S：固定surrogate backbone　A<sub>G</sub><sup>t</sup> / A<sub>G</sub><sup>t+1</sup>：集約前 / 後のglobal surrogate LoRA<br/>A<sub>k</sub><sup>t+1</sup>：client k の局所学習後LoRA　M<sub>G</sub><sup>t</sup> = S + A<sub>G</sub><sup>t</sup><br/>q<sup>pre</sup> / q<sup>post</sup>：同一sampleの局所学習前 / 後の出力確率　Δq：出力変化<br/>μ<sup>post</sup> / Δμ：q<sup>post</sup> / Δqの集約値　Q<sub>k,r</sub>：prototype　π<sub>k,r</sub>：frequency"]
        L1MAIN ~~~ LEGEND
    end

    classDef client fill:#EAF2FF,stroke:#2563A6,stroke-width:2px,color:#102A43;
    classDef process fill:#FFFFFF,stroke:#37474F,stroke-width:2px,color:#172B4D;
    classDef output fill:#F3E8FF,stroke:#7E57A8,stroke-width:2px,color:#2D1B4E;
    classDef keep fill:#FFF2F0,stroke:#B94A48,stroke-width:2px,color:#5B1A18;
    classDef state fill:#EAF7EA,stroke:#438A43,stroke-width:2px,color:#173A17;
    classDef legend fill:#F7F9FB,stroke:#90A4AE,stroke-width:1px,color:#263238;

    class PRE,LOCAL,POST,DELTA client;
    class AGG process;
    class PROTO,L2USE output;
    class KEEP keep;
    class MGNEXT state;
    class LEGEND legend;
    style L1MAIN fill:none,stroke:none
```

発表時の説明:

- 順序は`q_(k,i)^pre`計算、局所更新、同じsampleで`q_(k,i)^post`計算、`Δq_(k,i)`計算で固定する。
- Stage Iでは`q^(pre)`と`q^(post)`の実体をsoftmax probability vectorとして扱う。raw logitsとは呼ばない。
- sample単位の`q^(post)`・`Δq`をclient内でclass-wiseに集約し、frequency付き`Q_(k,r)`を作る。`π_(k,r)`はそのprototypeに含まれるsample数であり、FedAvgの重みではない。
- 現在の実装方針ではclassification headも局所更新し、LoRAとともに集約・再配布する。
- clientから送るのはLoRA更新、classification head更新、`Q_(k,r)`と識別情報である。privateデータ、raw frame、client内sample ID、独立した`q^pre`は送らない。

## 3. 第2層：代理データと出力確率のペアの生成，選択，調整

説明の中心: PDFの層名は維持しつつ、Stage Iでペアにする`μ^(post)`は出力確率として示す。残差経路と追加の品質指標は別々のLater Phaseに置く。

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"linear","nodeSpacing":32,"rankSpacing":48},"themeVariables":{"fontFamily":"Yu Gothic, Meiryo, sans-serif","fontSize":"21px","lineColor":"#455A64"}}}%%
flowchart TB
    subgraph LAYER2["第2層：代理データと出力確率のペアの生成，選択，調整（クラウド）"]
        direction TB
        subgraph L2MAIN[" "]
        direction LR
        INPUT["入力<br/>Q<sub>k,r</sub>・M<sub>G</sub><sup>t</sup>・M<sub>G</sub><sup>t+1</sup><br/>シードデータ D<sub>seed</sub>"]
        RETRIEVE["1. シード検索<br/>p<sub>G</sub><sup>t+1</sup>でμ<sub>k,r</sub><sup>post</sup>に近い<br/>s<sub>k,r</sub><sup>*</sup>を検索"]
        TRANSFORM["2. 変換<br/>x̃<sub>k,r</sub> = T<sub>φ</sub>(s<sub>k,r</sub><sup>*</sup>)"]
        OPTIMIZE["3. 代理データ生成<br/>L<sub>proxy</sub> = L<sub>match</sub> + L<sub>reg</sub>"]
        QUALITY["4. 再現誤差・品質評価<br/>Q<sub>inv</sub>を記録"]
        SELECT["5. Basic pathの重み付け<br/>ρ<sub>k,r</sub><sup>Uniform</sup> = π<sub>k,r</sub><br/>Q<sub>inv</sub>-basedは第II段階"]
        PAIR["6. 代理データ・出力確率ペア<br/>P<sub>k,r</sub> = (x̃<sub>k,r</sub>, μ<sub>k,r</sub><sup>post</sup>)<br/>第3層へ渡す"]
        RESIDUAL["Optional / Later Phase：残差経路<br/>閾値超過時だけclient固有LoRAを一時再構成<br/>proxyを再生成または再評価"]
        ADV["Optional / Later Phase：追加指標<br/>Q<sub>cons</sub>・Q<sub>novel</sub>・Q<sub>retain</sub><br/>proxy pairの選択・重み付けに使用"]

        INPUT --> RETRIEVE --> TRANSFORM --> OPTIMIZE --> QUALITY
        QUALITY -->|"再現誤差が閾値以下<br/>Basic path"| SELECT --> PAIR
        QUALITY -.->|"再現誤差が閾値超過"| RESIDUAL
        RESIDUAL -.->|"再生成・再評価"| QUALITY
        ADV -.->|"必要時"| SELECT
        end

        subgraph L2LEGEND["凡例"]
            direction LR
            L2LG1["Q<sub>k,r</sub>：prototype<br/>μ<sub>k,r</sub><sup>post</sup>：teacher出力確率<br/>M<sub>G</sub><sup>t</sup> / M<sub>G</sub><sup>t+1</sup>：連合集約前 / 後"]
            L2LG2["p<sub>G</sub><sup>t+1</sup>：連合集約後の予測<br/>D<sub>seed</sub>：server-side seed<br/>s<sub>k,r</sub><sup>*</sup>：検索seed　x̃<sub>k,r</sub>：変換後"]
            L2LG3["L<sub>proxy</sub>：代理データ生成目的<br/>Q<sub>inv</sub>：品質指標　ρ<sub>k,r</sub>：第3層の重み<br/>P<sub>k,r</sub>：第3層へ渡す代理ペア"]
            L2LG4["実線：Basic path<br/>破線・橙：Optional / Later Phase"]
            L2LG1 ~~~ L2LG2 ~~~ L2LG3 ~~~ L2LG4
        end
        L2MAIN ~~~ L2LEGEND
    end

    classDef input fill:#EAF2FF,stroke:#2563A6,stroke-width:2px,color:#102A43;
    classDef process fill:#FFFFFF,stroke:#37474F,stroke-width:2px,color:#172B4D;
    classDef output fill:#F3E8FF,stroke:#7E57A8,stroke-width:2px,color:#2D1B4E;
    classDef optional fill:#FFF4DC,stroke:#C68A00,stroke-width:2px,stroke-dasharray:7 5,color:#5A3D00;
    classDef legend fill:#F7F9FB,stroke:#90A4AE,stroke-width:1px,color:#263238;
    classDef legendItem fill:transparent,stroke:transparent,color:#263238;

    class INPUT output;
    class RETRIEVE,TRANSFORM,OPTIMIZE,QUALITY,SELECT process;
    class PAIR output;
    class RESIDUAL,ADV optional;
    class L2LG1,L2LG2,L2LG3,L2LG4 legendItem;
    style L2MAIN fill:none,stroke:none
    style L2LEGEND fill:#F7F9FB,stroke:#90A4AE,stroke-width:1px
```

発表時の説明:

- `M_G^t`と`M_G^(t+1)`の同じ代理データ上での出力差を使い、`Δμ_(k,r)`の方向を再現する。client固有の`Δμ_(k,r)`とproxy側のglobal round差は別の量である。
- 基本経路は、検索、変換、`L_proxy`による代理データ生成、再現誤差・品質評価、選択、`P_(k,r)`生成の順である。
- Phase I Basic pathの重みは`ρ_(k,r)^Uniform=π_(k,r)`とする。`Q_inv`は記録し、`Q_inv`-basedの重み付け比較は第II段階で行う。
- 再現誤差が閾値を超えた場合だけ残差経路へ分岐する。残差経路の導入条件、閾値、採否は未確定である。
- `Q_cons`・`Q_novel`・`Q_retain`は選択・重み付け用の追加指標であり、残差経路ではない。
- 第3層へ渡すteacher labelは`μ_(k,r)^post`であり、global surrogate自身の出力ではない。

## 4. 第3層：異種モデル間の知識蒸留

説明の中心: Phase Iは現在ラウンドの代理ペアでfoundation LoRAを更新し、その状態を次ラウンドへ戻す。replay・anchor・補助的蒸留は別のLater Phaseとして扱う。

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"linear","nodeSpacing":34,"rankSpacing":50},"themeVariables":{"fontFamily":"Yu Gothic, Meiryo, sans-serif","fontSize":"20px","lineColor":"#455A64"}}}%%
flowchart LR
    subgraph LAYER3["第3層：異種モデル間の知識蒸留（クラウド）"]
        direction LR
        CURRENT["現在roundの代理ペア<br/>D<sub>proxy</sub><sup>t</sup><br/>P<sub>k,r</sub>・ρ<sub>k,r</sub>"]
        BASE["round t の基盤モデル<br/>予測 p<sub>F</sub><br/>更新前の基盤LoRA"]
        UPDATE["Phase I Basic path<br/>L<sub>current</sub><sup>t</sup>でfoundation LoRAのみ更新<br/>classification headは固定"]
        UPDATED["更新後の基盤LoRA<br/>クラウドで保持"]

        CURRENT --> UPDATE
        BASE --> UPDATE --> UPDATED
        UPDATED -->|"次round"| BASE

        MEMORY["Phase III<br/>過去の代理ペア<br/>D<sub>replay</sub><sup>t</sup>"]
        ANCHOR["Phase III<br/>固定anchorデータ<br/>D<sub>anchor</sub>"]
        MEMUPDATE["Phase III<br/>重要なP<sub>k,r</sub>を選び<br/>D<sub>replay</sub>を更新"]
        AUX["Optional / Later Phase<br/>foundationからsurrogateへの<br/>補助的蒸留"]
        SURR["戻り先<br/>第1層のglobal surrogate側<br/>更新方法・周期は未確定"]

        MEMORY -.->|"L<sub>replay</sub><sup>t</sup>"| UPDATE
        ANCHOR -.->|"L<sub>an</sub><sup>t</sup>"| UPDATE
        UPDATED -.-> MEMUPDATE
        MEMUPDATE -.-> MEMORY
        UPDATED -.-> AUX -.-> SURR

        LEGEND["凡例<br/>D<sub>proxy</sub><sup>t</sup>：現在roundの代理ペア集合　P<sub>k,r</sub>：代理データとteacher出力確率のペア<br/>ρ<sub>k,r</sub>：各pairの重み　p<sub>F</sub>：foundation modelの予測　L<sub>current</sub><sup>t</sup>：現在roundの蒸留損失<br/>D<sub>replay</sub><sup>t</sup>：過去の代理ペア　D<sub>anchor</sub>：固定anchorデータ<br/>実線：Phase I Basic path　破線・橙：Phase III / Optional"]
    end

    classDef current fill:#EAF2FF,stroke:#2563A6,stroke-width:2px,color:#102A43;
    classDef phase3 fill:#FFF4DC,stroke:#C68A00,stroke-width:2px,stroke-dasharray:7 5,color:#5A3D00;
    classDef process fill:#FFFFFF,stroke:#37474F,stroke-width:2px,color:#172B4D;
    classDef model fill:#EAF7EA,stroke:#438A43,stroke-width:2px,color:#173A17;
    classDef output fill:#F3E8FF,stroke:#7E57A8,stroke-width:2px,color:#2D1B4E;
    classDef legend fill:#F7F9FB,stroke:#90A4AE,stroke-width:1px,color:#263238;

    class CURRENT output;
    class MEMORY,ANCHOR,MEMUPDATE,AUX,SURR phase3;
    class UPDATE process;
    class BASE,UPDATED model;
    class LEGEND legend;
```

発表時の説明:

- Phase Iでは`D_proxy^t`の`P_(k,r)`を使い、`L_current^t`でfoundation LoRAだけを更新する。現在の実装ではfoundation側classification headは固定する。
- 更新後の基盤LoRAは終了点ではなく、クラウドで保持して次ラウンドの第3層の開始状態へ戻す。
- Phase IIIでは過去の`D_replay^t`と`D_anchor`を追加し、更新後に重要な`P_(k,r)`を選んでreplay memoryを更新する。
- foundationからsurrogateへの補助的蒸留はLater Phaseである。残す場合は、更新後foundationから第1層のglobal surrogate側へ知識を還元する流れとして示す。
- surrogate LoRAとfoundation LoRAは独立しており、LoRA parameterを両model間で直接転送しない。

## PowerPointへ載せる際の制約

- 4図をそれぞれ別スライドに配置する。
- スライドタイトルは図の見出しをそのまま使用する。
- Mermaid画像はスライド横幅の80から90パーセントを使用する。
- 本文相当の文字は17pt未満にしない。
- 各スライドに図中の凡例を残す。
- 数式の詳細、候補ごとのaccept / reject、最新runの診断値は口頭説明か補足スライドへ移す。
- 青はclient側、紫は層間で渡すデータ、緑はモデル状態、赤はclient内保持、白は基本処理、橙の破線はOptional / Later Phaseとして統一する。
