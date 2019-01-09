========================
Sparse Bundle Adjustment
========================

----
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
- 誤差関数のヤコビ行列を計算する際に， :math:`\mathbb{so}(3)` や四元数などに関する微分が現れるため，手法が複雑である

--------
問題設定
--------

得たいもの
----------

- 3次元空間におけるランドマーク座標 :math:`\mathbf{b}_{j},j=1,\dots,n`
- カメラ姿勢 :math:`\mathbf{a}_{i} = [\mathbf{t}_{i}, \mathbf{\omega}_{i}],i=1,\dots,m`
  ただし :math:`\mathbf{t}` は並進，:math:`\mathbf{\omega}` は :math:`\mathbb{so}(3)` の元である．

入力
----

各視点から観測されたランドマークの像の集合 :math:`\mathbf{X}`

.. math::
    \mathbf{X} = \begin{bmatrix}
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


----
目的
----

投影モデルを :math:`\mathbf{Q}(\mathbf{a}_{i},\mathbf{b}_{j})` とし，以下の誤差関数を最小化するような :math:`\mathbf{P} = \left[ \mathbf{a}^{\top}_{1}, \dots, \mathbf{a}^{\top}_{m}, \mathbf{b}^{\top}_{1}, \dots, \mathbf{b}^{\top}_{n} \right]` を求める．

.. math::
    E(\mathbf{P}) = \begin{align}
    \sum_{i=1}^{n} \sum_{j=1}^{m} d_{\Sigma_{\mathbf{x}_{ij}}}(\mathbf{Q}(\mathbf{a}_{j}, \mathbf{b}_{i}), \mathbf{x}_{ij})^{2}
    \end{align}


ここで :math:`d_{\Sigma_{\mathbf{x}}}` は

.. math::
    d_{\Sigma_{\mathbf{x}}}(\mathbf{x}_{1}, \mathbf{x}_{2}) =
    \sqrt{(\mathbf{x}_{1} - \mathbf{x}_{2})^{\top} \Sigma^{-1}_{\mathbf{x}} (\mathbf{x}_{1} - \mathbf{x}_{2})}

で定義される距離関数である．

.. math::
    \begin{align}
    \hat{\mathbf{X}} &= \begin{bmatrix}
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
    \end{bmatrix} \\
    \hat{\mathbf{x}}_{ij} &= \mathbf{Q}(\mathbf{a}_{j}, \mathbf{b}_{i}) \\
    \Sigma_{\mathbf{X}} &= diag(\Sigma_{\mathbf{x}_{11}}, \dots, \Sigma_{\mathbf{x}_{1m}},
                                \Sigma_{\mathbf{x}_{21}}, \dots, \Sigma_{\mathbf{x}_{2m}},
                                \dots,
                                \Sigma_{\mathbf{x}_{n1}}, \dots, \Sigma_{\mathbf{x}_{nm}})
    \end{align}

とおけば，誤差を次のように表現することができる．

.. math::
    E(\mathbf{P}) = (\mathbf{X}-\hat{\mathbf{X}})^{\top} \Sigma_{\mathbf{X}} (\mathbf{X}-\hat{\mathbf{X}})

----------
解法の概要
----------

SBAでは誤差関数を LM法_ によって最小化する．

.. _LM法: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm


:math:`\mathbf{P}` の更新量を :math:`\delta_{\mathbf{P}}` とする．

.. math::
    \mathbf{P}_{t+1} \leftarrow \mathbf{P}_{t} + \delta_{\mathbf{P}}

この更新量 :math:`\delta_{\mathbf{P}}` は次の線形方程式を解くことによって得られる．

.. math::
    (\mathbf{J}^{\top}\mathbf{J} + \lambda \mathbf{I}) \delta_{\mathbf{P}}
    = \mathbf{J}^{\top} (\mathbf{X} - \hat{\mathbf{X}}) \\
    :label: lm_update

ここで :math:`\mathbf{J} = \frac{\partial \hat{\mathbf{X}}}{\partial \mathbf{P}}` である．

SBAは，このヤコビ行列 :math:`\mathbf{J}` がスパースであることに着目し，計算量を削減している．


ヤコビ行列のスパース性
----------------------

:math:`\forall j \neq k` について

.. math::
    \frac{\partial \mathbf{Q}(\mathbf{a}_{j}, \mathbf{b}_{i})}{\partial \mathbf{a}_{k}} = \mathbf{0}

:math:`\forall i \neq k` について

.. math::
    \frac{\partial \mathbf{Q}(\mathbf{a}_{j}, \mathbf{b}_{i})}{\partial \mathbf{b}_{k}} = \mathbf{0}

が成り立つことから，ヤコビ行列 :math:`\mathbf{J}` はスパースな行列になる．
この性質を利用すると，:eq:`lm_update` のうち必要な部分のみを計算することで効率よく :math:`\delta_{\mathbf{P}}` を求めることが可能となる．

線型方程式の変形
----------------

:math:`A_{ij}=\frac{\partial \mathbf{Q}(\mathbf{a}_{j}, \mathbf{a}_{j})}{\partial \mathbf{a}_{j}}` ，
:math:`B_{ij}=\frac{\partial \mathbf{Q}(\mathbf{a}_{j}, \mathbf{b}_{i})}{\partial \mathbf{b}_{i}}`
とおくと，:math:`n=4` ，:math:`m=3` のとき， :math:`\mathbf{J}` は

.. math::
    \mathbf{J} = \begin{bmatrix}
        \mathbf{A}_{11} & \mathbf{0} & \mathbf{0} & \mathbf{B}_{11} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
        \mathbf{0} & \mathbf{A}_{11} & \mathbf{0} & \mathbf{B}_{12} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
        \mathbf{0} & \mathbf{0} & \mathbf{A}_{11} & \mathbf{B}_{13} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
        \mathbf{A}_{21} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{B}_{21} & \mathbf{0} & \mathbf{0} \\
        \mathbf{0} & \mathbf{A}_{21} & \mathbf{0} & \mathbf{0} & \mathbf{B}_{22} & \mathbf{0} & \mathbf{0} \\
        \mathbf{0} & \mathbf{0} & \mathbf{A}_{21} & \mathbf{0} & \mathbf{B}_{23} & \mathbf{0} & \mathbf{0} \\
        \mathbf{A}_{31} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{B}_{31} & \mathbf{0} \\
        \mathbf{0} & \mathbf{A}_{31} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{B}_{32} & \mathbf{0} \\
        \mathbf{0} & \mathbf{0} & \mathbf{A}_{31} & \mathbf{0} & \mathbf{0} & \mathbf{B}_{33} & \mathbf{0} \\
        \mathbf{A}_{41} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{B}_{41} \\
        \mathbf{0} & \mathbf{A}_{41} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{B}_{42} \\
        \mathbf{0} & \mathbf{0} & \mathbf{A}_{41} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{B}_{43} \\
    \end{bmatrix}

となる．

Appendix
--------

LM法の導出
----------

:math:`\Sigma` を分散共分散行列とし，誤差をmahalanobis距離によって次のように定義する．

.. math::
    d_{\Sigma}(\mathbf{y}, \mathbf{f}(\mathbf{\beta} + \mathbf{\delta})) = (\mathbf{y} - \mathbf{f}(\mathbf{\beta} + \mathbf{\delta}))^{\top}\Sigma (\mathbf{y} - \mathbf{f}(\mathbf{\beta} + \mathbf{\delta}))
    :label: lm_error


関数 :math:`\mathbf{f}` を :math:`\mathbf{f}(\mathbf{\beta} + \mathbf{\delta}) \approx \mathbf{f}(\mathbf{\beta}) + J \mathbf{\delta},J=\frac{\partial \mathbf{f}}{\partial \beta}` と近似すると， :eq:`lm_error` は

.. math::
    \begin{align}
    d_{\Sigma}(\mathbf{y}, \mathbf{f}(\mathbf{\beta} + \mathbf{\delta}))
    &\approx (\mathbf{y} - \mathbf{f}(\mathbf{\beta}) - \mathbf{J}\mathbf{\delta})^{\top} \Sigma (\mathbf{y} - \mathbf{f}(\mathbf{\beta}) - \mathbf{J}\mathbf{\delta}) \\
    &= (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))^{\top} \Sigma  (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))
    - 2 (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))^{\top} \Sigma^{-1} \mathbf{J} \mathbf{\delta}
    + \mathbf{\delta}^{\top} \mathbf{J}^{\top} \Sigma^{-1} \mathbf{J} \mathbf{\delta}
    \end{align}

となる．


:eq:`lm_error` を

.. math::
    E(p) = (\mathbf{X}-\hat{\mathbf{X}})^{\top} \Sigma_{\mathbf{X}} (\mathbf{X}-\hat{\mathbf{X}})


