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

SBAでは誤差関数を LM法_ [#Levenberg_1944]_ によって最小化する．
さらに，LM法に現れるヤコビ行列の構造に着目し，LM法における更新量の計算を複数の線型方程式に分解することで，計算量を削減している．

.. _LM法: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm


:math:`\mathrm{P}` の更新量を :math:`\delta_{\mathrm{P}}` とする．

.. math::
    \mathrm{P}_{t+1} \leftarrow \mathrm{P}_{t} + \delta_{\mathrm{P}}

この更新量 :math:`\delta_{\mathrm{P}}` は次の線形方程式を解くことによって得られる．

.. math::
    (\mathrm{J}^{\top}\mathrm{J} + \lambda \mathrm{I}) \delta_{\mathrm{P}}
    = \mathrm{J}^{\top} (\mathrm{X} - \hat{\mathrm{X}}) \\
    :label: lm_update

ここで :math:`\mathrm{J} = \frac{\partial \hat{\mathrm{X}}}{\partial \mathrm{P}}` である．

SBAは，このヤコビ行列 :math:`\mathrm{J}` がスパースであることに着目し，計算量を削減している．


ヤコビ行列のスパース性
~~~~~~~~~~~~~~~~~~~~~~

:math:`\forall j \neq k` について

.. math::
    \frac{\partial \mathrm{Q}(\mathbf{a}_{j}, \mathbf{b}_{i})}{\partial \mathbf{a}_{k}} = \mathbf{0}

:math:`\forall i \neq k` について

.. math::
    \frac{\partial \mathrm{Q}(\mathbf{a}_{j}, \mathbf{b}_{i})}{\partial \mathbf{b}_{k}} = \mathbf{0}

が成り立つことから，ヤコビ行列 :math:`\mathrm{J}` はスパースな行列になる．
この性質を利用すると，:eq:`lm_update` のうち必要な部分のみを計算することで効率よく :math:`\delta_{\mathrm{P}}` を求めることが可能となる．


例
~~


:math:`\mathrm{A}_{ij}=\frac{\partial \mathrm{Q}(\mathbf{a}_{j}, \mathbf{b}_{i})}{\partial \mathbf{a}_{j}}` ，
:math:`\mathrm{B}_{ij}=\frac{\partial \mathrm{Q}(\mathbf{a}_{j}, \mathbf{b}_{i})}{\partial \mathbf{b}_{i}}`
とおくと，:math:`n=4` ，:math:`m=3` のとき， :math:`\mathrm{J}` は

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



.. [#Gallego_et_al_2015] Gallego, Guillermo, and Anthony Yezzi. "A compact formula for the derivative of a 3-D rotation in exponential coordinates." Journal of Mathematical Imaging and Vision 51.3 (2015): 378-384.
.. [#Levenberg_1944] Levenberg, Kenneth. "A method for the solution of certain non-linear problems in least squares." Quarterly of applied mathematics 2.2 (1944): 164-168.
