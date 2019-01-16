========================
Sparse Bundle Adjustment
========================

概要
----

Sparse Bundle Adjustment(SBA)はSfMの手法の一種である．
SfMは一般的に再投影誤差の最小化問題として記述できるが，推定の対象となるランドマークの数や各視点におけるカメラ姿勢が膨大になり，計算コストが非常に大きくなってしまう．
SBAは再投影誤差のヤコビ行列がスパースになることに着目し，計算量を大幅に削減した手法である．

Tomasi-Kanade法と比較すると，次のような特徴がある．

利点

- ある視点からは見えないランドマークがあったとしても復元を行うことができる
- 内部行列を指定できるため，正投影以外にも多様なカメラモデルを用いることができる

欠点

- 問題に応じてハイパーパラメータを調整しなければならない
- 誤差関数のヤコビ行列を計算する際に， :math:`\mathfrak{so}(3)` や四元数などに関する微分が現れるため，手法が複雑である


問題設定
--------

得たいもの
~~~~~~~~~~


- 3次元空間におけるランドマーク座標 :math:`\mathbf{b}_{j},j=1,\dots,n`
- カメラ姿勢 :math:`\mathbf{a}_{i} = [\mathbf{t}_{i}, \mathbf{\omega}_{i}],i=1,\dots,m`
  ただし :math:`\mathbf{t} \in \mathbb{R}^{3}` は並進を表すベクトルであり，:math:`\mathbf{\omega} \in \mathfrak{so}(3)` はカメラの回転を表す回転行列 :math:`R \in \mathbb{R}^{3 \times 3}` に対応するリー代数の元である．


入力
~~~~


各視点から観測されたランドマークの像の集合 :math:`\mathrm{X}`

.. math::
    \mathrm{X} = \begin{bmatrix}
        \mathbf{x}^{\top}_{11},
        \dots,
        \mathbf{x}^{\top}_{1m},
        \mathbf{x}^{\top}_{21},
        \dots,
        \mathbf{x}^{\top}_{2m},
        \dots,
        \mathbf{x}^{\top}_{n1},
        \dots,
        \mathbf{x}^{\top}_{nm}
    \end{bmatrix}


目的
----

投影モデルを :math:`\mathrm{Q}(\mathbf{a}_{i},\mathbf{b}_{j})` とし，以下の誤差関数を最小化するような :math:`\mathrm{P} = \left[\mathbf{a}, \mathbf{b}\right] = \left[ \mathbf{a}^{\top}_{1}, \dots, \mathbf{a}^{\top}_{m}, \mathbf{b}^{\top}_{1}, \dots, \mathbf{b}^{\top}_{n} \right]` を求める．

.. math::
    E(\mathrm{P}) = \begin{align}
    \sum_{i=1}^{n} \sum_{j=1}^{m} d_{\mathrm{\Sigma}_{\mathbf{x}_{ij}}}(\mathrm{Q}(\mathbf{a}_{j}, \mathbf{b}_{i}), \mathbf{x}_{ij})^{2}
    \end{align}


ここで :math:`d_{\mathrm{\Sigma}_{\mathbf{x}}}` は :math:`\mathbf{x}` に対応する分散共分散行列を :math:`\mathrm{\Sigma}_{\mathbf{x}}` として

.. math::
    d_{\mathrm{\Sigma}_{\mathbf{x}}}(\mathbf{x}_{1}, \mathbf{x}_{2}) =
    \sqrt{(\mathbf{x}_{1} - \mathbf{x}_{2})^{\top} \mathrm{\Sigma}^{-1}_{\mathbf{x}} (\mathbf{x}_{1} - \mathbf{x}_{2})}

で定義される距離関数である．

.. math::
    \begin{align}
    \hat{\mathrm{X}} &= \begin{bmatrix}
        \hat{\mathbf{x}}^{\top}_{11},
        \dots,
        \hat{\mathbf{x}}^{\top}_{1m},
        \hat{\mathbf{x}}^{\top}_{21},
        \dots,
        \hat{\mathbf{x}}^{\top}_{2m},
        \dots,
        \hat{\mathbf{x}}^{\top}_{n1},
        \dots,
        \hat{\mathbf{x}}^{\top}_{nm}
    \end{bmatrix}^{\top} \\
    \hat{\mathbf{x}}_{ij}
    &= \mathrm{Q}(\mathbf{a}_{j}, \mathbf{b}_{i}) \\
    \mathrm{\Sigma}_{\mathrm{X}}
    &= diag(
        \mathrm{\Sigma}_{\mathbf{x}_{11}},
        \dots,
        \mathrm{\Sigma}_{\mathbf{x}_{1m}},
        \mathrm{\Sigma}_{\mathbf{x}_{21}},
        \dots,
        \mathrm{\Sigma}_{\mathbf{x}_{2m}},
        \dots,
        \mathrm{\Sigma}_{\mathbf{x}_{n1}},
        \dots,
        \mathrm{\Sigma}_{\mathbf{x}_{nm}}
    )
    \end{align}

とおけば，誤差を次のように表現することができる．

.. math::
    E(\mathrm{P}) = (\mathrm{X}-\hat{\mathrm{X}})^{\top} \mathrm{\Sigma}_{\mathrm{X}} (\mathrm{X}-\hat{\mathrm{X}})


解法の概要
----------

SBAでは，誤差関数を最小化するような :math:`\mathrm{P}` を見つけるため， :math:`\mathrm{P}^{(t)}` を逐次的に更新し，誤差関数を探索する．すなわち，時刻 :math:`t` における :math:`\mathrm{P}` の更新量を :math:`\delta_{\mathrm{P}}^{(t)} = \left[ \delta_{\mathbf{a}_{1}}^{\top}, \dots, \delta_{\mathbf{a}_{m}}^{\top}, \delta_{\mathbf{b}_{1}}^{\top}, \dots, \delta_{\mathbf{b}_{n}}^{\top} \right]` ` として，

.. math::
    \mathrm{P}^{(t+1)} \leftarrow \mathrm{P}^{(t)} + \delta_{\mathrm{P}}^{(t)}

というふうに :math:`\mathrm{P}^{(t)}` を更新することで誤差関数を最小化するような :math:`\mathrm{P}` を見つける．

更新量 :math:`\delta_{\mathrm{P}}^{(t)}` の計算にはLM法_ [#Levenberg_1944]_ を用いる．さらに，LM法に現れるヤコビ行列の構造に着目し，更新量の計算を複数の線型方程式に分解することで，計算量を削減している．

.. _LM法: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

LM法を用いる場合，この更新量 :math:`\delta_{\mathrm{P}}` は次の線型方程式を解くことによって得られる．

.. math::
    \left[
        \mathrm{J}^{\top} \mathrm{\mathrm{\Sigma}}^{-1} \mathrm{J} + \lambda \mathrm{I}
    \right]
    \delta_{\mathrm{P}}^{(t)}
    = \mathrm{J}^{\top} \mathrm{\mathrm{\Sigma}}^{-1} \left[ \mathrm{X} - \hat{\mathrm{X}} \right] \\
    :label: lm-update

:math:`\mathbf{J}` は :math:`\hat{\mathrm{X}}` のヤコビ行列 :math:`\mathrm{J} = \frac{\partial \hat{\mathrm{X}}}{\partial \mathrm{P}} \rvert_{\mathrm{P}=\mathrm{P}^{(t)}}` であり， :math:`\lambda \in \mathbb{R}, \lambda \geq 0` は damping parameter である．

SBAでは，:math:`\mathrm{J}` の構造に着目し， :eq:`lm-update` をより小さい複数の線型方程式に分解することで，計算を高速化している．


線型方程式の分解
~~~~~~~~~~~~~~~~

まず :math:`\mathrm{J}` を分解する． :math:`\mathrm{P}` の定義より，

.. math::
    \mathrm{A} = \frac{\partial \hat{\mathrm{X}}}{\partial \mathbf{a}},
    \mathrm{B} = \frac{\partial \hat{\mathrm{X}}}{\partial \mathbf{b}}

とおけば， :math:`\mathrm{J}` は

.. math::
    \mathrm{J} = \frac{\partial \hat{\mathrm{X}}}{\partial \mathrm{P}}
    = \frac{\partial \hat{\mathrm{X}}}{\partial (\mathrm{a}, \mathrm{b})} = \left[ A, B \right]
    :label: decomposition-J

と書ける．

次に :eq:`lm-update` の右辺を分解する． :eq:`decomposition-J` を用いると， :eq:`lm-update` の右辺は

.. math::
    \begin{align}
        \mathbf{\epsilon}_{\mathbf{a}} &= A^{\top} \mathrm{\Sigma}^{-1} (\mathrm{X} - \hat{\mathrm{X}}) \\
        \mathbf{\epsilon}_{\mathbf{b}} &= B^{\top} \mathrm{\Sigma}^{-1} (\mathrm{X} - \hat{\mathrm{X}})
    \end{align}

とおくことによって，

.. math::
    \mathrm{J}^{\top} \mathrm{\mathrm{\Sigma}}^{-1} (\mathrm{X} - \hat{\mathrm{X}})
    = \begin{bmatrix} \mathbf{\epsilon}_{\mathbf{a}} \\ \mathbf{\epsilon}_{\mathbf{b}} \end{bmatrix}

と書ける．

さらに :eq:`lm-update` の左辺を分解する．
左辺の :math:`\mathrm{J}^{\top} \mathrm{\mathrm{\Sigma}}^{-1} \mathrm{J}` という項は大きく4つの行列に分解することができる．

.. math::
    \begin{align}
        \mathrm{J}^{\top} \mathrm{\mathrm{\Sigma}}^{-1} \mathrm{J}
        &= \begin{bmatrix}
            A^{\top} \\ B^{\top}
        \end{bmatrix}
        \mathrm{\Sigma}^{-1}
        \begin{bmatrix}
            A & B
        \end{bmatrix} \\
        &= \begin{bmatrix}
            A^{\top} \mathrm{\Sigma}^{-1} A & A^{\top} \mathrm{\Sigma}^{-1} B \\
            B^{\top} \mathrm{\Sigma}^{-1} A & B^{\top} \mathrm{\Sigma}^{-1} B
        \end{bmatrix} \\
        &= \begin{bmatrix}
            \mathrm{U} & \mathrm{W} \\
            \mathrm{W}^{\top} & \mathrm{V}
        \end{bmatrix}
    \end{align}

以上の結果を用いると， :eq:`lm-update` は

.. math::
    \left[
    \begin{bmatrix}
        \mathrm{U} & \mathrm{W} \\
        \mathrm{W}^{\top} & \mathrm{V}
    \end{bmatrix}
    +
    \begin{bmatrix}
        \lambda \mathrm{I} & \mathrm{0} \\
        \mathrm{0} & \lambda \mathrm{I}
    \end{bmatrix}
    \right]
    \begin{bmatrix}
        \mathbf{\delta}_{\mathbf{a}} \\
        \mathbf{\delta}_{\mathbf{b}}
    \end{bmatrix}
    =
    \begin{bmatrix}
        \mathbf{\epsilon}_{\mathbf{a}} \\
        \mathbf{\epsilon}_{\mathbf{b}}
    \end{bmatrix}

という形にすることができる．
さらに，

.. math::
    \begin{align}
        \mathrm{U}^{*} &= \mathrm{U} + \lambda \mathrm{I} \\
        \mathrm{V}^{*} &= \mathrm{V} + \lambda \mathrm{I}
    \end{align}

とおけば，

.. math::
    \begin{bmatrix}
        \mathrm{U}^{*} & \mathrm{W} \\
        \mathrm{W}^{\top} & \mathrm{V}^{*}
    \end{bmatrix}
    \begin{bmatrix}
        \mathbf{\delta}_{\mathbf{a}} \\
        \mathbf{\delta}_{\mathbf{b}}
    \end{bmatrix}
    =
    \begin{bmatrix}
        \mathbf{\epsilon}_{\mathbf{a}} \\
        \mathbf{\epsilon}_{\mathbf{b}}
    \end{bmatrix}

となる．この両辺に

.. math::
    \begin{bmatrix}
        \mathrm{I} & -\mathrm{W}{\mathrm{V}^{*}}^{-1} \\
        \mathrm{0} & \mathrm{I}
    \end{bmatrix}

という行列を左から作用させると，

.. math::
    \begin{bmatrix}
        \mathrm{I} & -\mathrm{W}{\mathrm{V}^{*}}^{-1} \\
        \mathrm{0} & \mathrm{I}
    \end{bmatrix}
    \begin{bmatrix}
        \mathrm{U}^{*} & \mathrm{W} \\
        \mathrm{W}^{\top} & \mathrm{V}^{*}
    \end{bmatrix}
    \begin{bmatrix}
        \mathbf{\delta}_{\mathbf{a}} \\
        \mathbf{\delta}_{\mathbf{b}}
    \end{bmatrix}
    =
    \begin{bmatrix}
        \mathrm{I} & -\mathrm{W}{\mathrm{V}^{*}}^{-1} \\
        \mathrm{0} & \mathrm{I}
    \end{bmatrix}
    \begin{bmatrix}
        \mathbf{\epsilon}_{\mathbf{a}} \\
        \mathbf{\epsilon}_{\mathbf{b}}
    \end{bmatrix} \\
    :label: left-multiplication

.. math::
    \begin{bmatrix}
        \mathrm{U}^{*} - \mathrm{W}{\mathrm{V}^{*}}^{-1}\mathrm{W}^{\top} & \mathrm{0} \\
        \mathrm{W}^{\top} & \mathrm{V}^{*}
    \end{bmatrix}
    \begin{bmatrix}
        \mathbf{\delta}_{\mathbf{a}} \\
        \mathbf{\delta}_{\mathbf{b}}
    \end{bmatrix}
    =
    \begin{bmatrix}
        \mathbf{\epsilon}_{\mathbf{a}} - \mathrm{W}{\mathrm{V}^{*}}^{-1}\mathbf{\epsilon}_{\mathbf{b}} \\
        \mathbf{\epsilon}_{\mathbf{b}}
    \end{bmatrix}
    :label: affected-from-left

という形にすることができる．ここから2つの方程式を取り出す．
すると， :eq:`affected-from-left` において左辺の行列の右上が :math:`\mathrm{0}` になったことから， :math:`\mathbf{\delta}_{\mathbf{a}}` についての式 :eq:`derivation-delta-a` を得ることができる．

.. math::
    (\mathrm{U}^{*} - \mathrm{W}{\mathrm{V}^{*}}^{-1}\mathrm{W}^{\top}) \mathbf{\delta}_{\mathbf{a}}
    = \mathbf{\epsilon}_{\mathbf{a}} - \mathrm{W}{\mathrm{V}^{*}}^{-1}\mathbf{\epsilon}_{\mathbf{b}}
    :label: derivation-delta-a

.. math::
    \mathrm{V}^{*} \mathbf{\delta}_{\mathbf{b}}
    = \mathbf{\epsilon}_{\mathbf{b}} - \mathrm{W}^{\top} \mathbf{\delta}_{\mathbf{a}}
    :label: derivation-delta-b

したがって，:eq:`derivation-delta-a` を先に解き，得られた :math:`\mathbf{\delta}_{\mathbf{a}}` を :eq:`derivation-delta-b` に代入すれば :math:`\mathbf{\delta}_{\mathbf{b}}` を得ることができる．


計算量の削減
~~~~~~~~~~~~

問題のサイズ(視点数や復元対象となるランドマークの数)が大きいときは， :eq:`lm-update` を直接解いて :math:`\mathbf{\delta}_{\mathrm{P}}` を得るよりも， :eq:`derivation-delta-a` と :eq:`derivation-delta-b` によって :math:`\mathbf{\delta}_{\mathbf{a}}` と :math:`\mathbf{\delta}_{\mathbf{b}}` をそれぞれ計算し結合することで :math:`\mathbf{\delta}_{\mathrm{P}}` を得た方が圧倒的に高速である．

| :eq:`lm-update` ， :eq:`derivation-delta-a` ， :eq:`derivation-delta-b` はいずれも線型方程式 :math:`\mathbf{y} = \mathrm{A}\mathbf{x},\; \mathbf{x} \in \mathbb{R}^{n}, \mathbf{y} \in \mathbb{R}^{m}, \mathrm{A} \in \mathbb{R}^{n \times m}` のかたちをしているため，:eq:`lm-update` から直接 :math:`\mathbf{\delta}_{\mathrm{P}}` を得る場合と， :eq:`derivation-delta-a` ， :eq:`derivation-delta-b` をそれぞれ解いて :math:`\mathbf{\delta}_{\mathrm{P}}` を得る場合のどちらも線型方程式を解くことになる．
| 線型方程式の解は :math:`\mathbf{x} = (\mathrm{A}^{\top}\mathrm{A})^{-1}\mathrm{A}^{\top}\mathbf{y}` を解くことで得られるが， :math:`n \times n` 行列の逆行列の計算は :math:`O(n^{2.3})` 〜 :math:`O(n^{3})` 程度のオーダーとなってしまう．
  すなわち，問題のサイズが大きくなると計算量が急激に増加するため，大きな問題を直接解くよりも，大きな問題を複数の小さな問題に分割して解いた方が計算コストを抑えることができる．
| SBAでは，式 :eq:`lm-update` を直接解く代わりに，それを小さく分割して得た :eq:`derivation-delta-a` と :eq:`derivation-delta-b` をそれぞれ解くことによって，計算コストを削減している．


ヤコビ行列のスパース性
~~~~~~~~~~~~~~~~~~~~~~
ヤコビ行列 :math:`\mathrm{J}` はスパースな行列になる．これは，:math:`\forall j \neq k` について

.. math::
    \frac{\partial \mathrm{Q}(\mathbf{a}_{j}, \mathbf{b}_{i})}{\partial \mathbf{a}_{k}} = \mathbf{0}

:math:`\forall i \neq k` について

.. math::
    \frac{\partial \mathrm{Q}(\mathbf{a}_{j}, \mathbf{b}_{i})}{\partial \mathbf{b}_{k}} = \mathbf{0}

が成り立つためである．


例えば，:math:`n=4` ，:math:`m=3` のとき，
:math:`\mathrm{A}_{ij}=\frac{\partial \mathrm{Q}(\mathbf{a}_{j}, \mathbf{b}_{i})}{\partial \mathbf{a}_{j}}` ，
:math:`\mathrm{B}_{ij}=\frac{\partial \mathrm{Q}(\mathbf{a}_{j}, \mathbf{b}_{i})}{\partial \mathbf{b}_{i}}`
とおけば，:math:`\mathrm{J}` は

.. math::
    \mathrm{J} = \begin{bmatrix}
        \mathrm{A}_{11} & \mathbf{0} & \mathbf{0} & \mathrm{B}_{11} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
        \mathbf{0} & \mathrm{A}_{11} & \mathbf{0} & \mathrm{B}_{12} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
        \mathbf{0} & \mathbf{0} & \mathrm{A}_{11} & \mathrm{B}_{13} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
        \mathrm{A}_{21} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathrm{B}_{21} & \mathbf{0} & \mathbf{0} \\
        \mathbf{0} & \mathrm{A}_{21} & \mathbf{0} & \mathbf{0} & \mathrm{B}_{22} & \mathbf{0} & \mathbf{0} \\
        \mathbf{0} & \mathbf{0} & \mathrm{A}_{21} & \mathbf{0} & \mathrm{B}_{23} & \mathbf{0} & \mathbf{0} \\
        \mathrm{A}_{31} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathrm{B}_{31} & \mathbf{0} \\
        \mathbf{0} & \mathrm{A}_{31} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathrm{B}_{32} & \mathbf{0} \\
        \mathbf{0} & \mathbf{0} & \mathrm{A}_{31} & \mathbf{0} & \mathbf{0} & \mathrm{B}_{33} & \mathbf{0} \\
        \mathrm{A}_{41} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathrm{B}_{41} \\
        \mathbf{0} & \mathrm{A}_{41} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathrm{B}_{42} \\
        \mathbf{0} & \mathbf{0} & \mathrm{A}_{41} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathrm{B}_{43} \\
    \end{bmatrix}

となる．

計算の簡略化
------------




勾配の具体的な計算方法
----------------------

SBAでは再投影誤差を勾配ベースの最適化手法で最小化することで姿勢パラメータ :math:`\mathbf{a}` と3次元点の座標 :math:`\mathbf{b}` を求めているため，画像平面に投影された像 :math:`\hat{\mathbf{x}}` の :math:`\mathbf{a}` と :math:`\mathbf{b}` それぞれについての微分を計算する必要がある．


姿勢パラメータに関する微分
~~~~~~~~~~~~~~~~~~~~~~~~~~


姿勢パラメータ :math:`\mathbf{a} = \left[ \mathbf{t}, \mathbf{\omega} \right]` に関する微分 :math:`\mathrm{A}=\frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{a}} =\begin{bmatrix} \frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{t}} & \frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{\omega}} \end{bmatrix}` は次のようになる．


.. math::
    \begin{align}
    \frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{t}}
    &= \frac{\partial \pi(\mathbf{p})}{\partial \mathbf{p}}
       \bigg\rvert_{\mathbf{p}=\mathrm{K}(\mathrm{R}\mathbf{b} + \mathbf{t})}
       \cdot
       \mathrm{K}
       \cdot
       \frac{\partial (\mathrm{R}(\mathbf{\omega})\mathbf{b} + \mathbf{v})}{\partial \mathbf{v}}
       \bigg\rvert_{\mathbf{v}=\mathbf{t}} \\
    &= \frac{\partial \pi(\mathbf{p})}{\partial \mathbf{p}}
       \bigg\rvert_{\mathbf{p}=\mathrm{K}(\mathrm{R}\mathbf{b} + \mathbf{t})}
       \cdot
       \mathrm{K}
    \end{align}


.. math::
    \begin{align}
    \frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{\omega}}
    &= \frac{\partial \pi(\mathbf{p})}{\partial \mathbf{p}}
       \bigg\rvert_{\mathbf{p}=\mathrm{K}(\mathrm{R}\mathbf{b} + \mathbf{t})}
       \cdot
       \mathrm{K}
       \cdot
       \frac{\partial (\mathrm{R}(\mathbf{v})\mathbf{b} + \mathbf{t})}{\partial \mathbf{v}}
       \bigg\rvert_{\mathbf{v}=\mathbf{\omega}} \\
    &= \frac{\partial \pi(\mathbf{p})}{\partial \mathbf{p}}
       \bigg\rvert_{\mathbf{p}=\mathrm{K}(\mathrm{R}\mathbf{b} + \mathbf{t})}
       \cdot
       \mathrm{K}
       \cdot
       \frac{\partial (\mathrm{R}(\mathbf{v})\mathbf{b})}{\partial \mathbf{v}}
       \bigg\rvert_{\mathbf{v}=\mathbf{\omega}}
    \end{align}


ここで， :math:`\frac{\partial (\mathrm{R}(\mathbf{v})\mathbf{b})}{\partial \mathbf{v}}` は [#Gallego_et_al_2015]_ による計算結果を用いることができる

.. math::
   \frac{\partial (\mathrm{R}(\mathbf{v})\mathbf{b})}{\partial \mathbf{v}}
   = -\mathrm{R}(\mathbf{v}) \left[ \mathbf{b} \right]_{\times}
     \frac{
        \mathbf{v}\mathbf{v}^{\top} +
        (\mathrm{R}(\mathbf{v})^{\top} - \mathrm{I}) \left[ \mathbf{v} \right]_{\times}
     }{||\mathbf{v}||^{2}}


3次元点座標に関する微分
~~~~~~~~~~~~~~~~~~~~~~~

3次元点の座標 :math:`\mathbf{b}` に関する微分 :math:`\mathrm{B}=\frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{b}}` は次のようになる．

.. math::
    \begin{align}
    \frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{b}}
    &= \frac{\partial \pi(\mathbf{p})}{\partial \mathbf{p}}
       \bigg\rvert_{\mathbf{p}=\mathrm{K}(\mathrm{R}\mathbf{b} + \mathbf{t})}
       \cdot
       \mathrm{K}
       \cdot
       \frac{\partial (\mathrm{R}(\mathbf{\omega})\mathbf{v} + \mathbf{t})}{\partial \mathbf{v}}
       \bigg\rvert_{\mathbf{v}=\mathbf{b}} \\
    &= \frac{\partial \pi(\mathbf{p})}{\partial \mathbf{p}}
       \bigg\rvert_{\mathbf{p}=\mathrm{K}(\mathrm{R}\mathbf{b} + \mathbf{t})}
       \cdot
       \mathrm{K}
       \cdot
       \mathrm{R}(\mathbf{\omega})
    \end{align}



PCG法による更新
---------------


.. [#Gallego_et_al_2015] Gallego, Guillermo, and Anthony Yezzi. "A compact formula for the derivative of a 3-D rotation in exponential coordinates." Journal of Mathematical Imaging and Vision 51.3 (2015): 378-384.
.. [#Levenberg_1944] Levenberg, Kenneth. "A method for the solution of certain non-linear problems in least squares." Quarterly of applied mathematics 2.2 (1944): 164-168.
