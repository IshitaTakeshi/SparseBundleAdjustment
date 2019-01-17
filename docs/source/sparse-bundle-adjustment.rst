========================
Sparse Bundle Adjustment
========================


.. math::
    \def\Cov{{\mathrm{\Sigma}}}
    \def\D{{\delta}}
    \def\VStar{{\mathrm{V}^{*}}}
    \def\Db{{\bf{\D}_{\bf{b}}}}
    \def\Da{{\bf{\D}_{\bf{a}}}}


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


- 3次元空間におけるランドマーク座標 :math:`\bf{b}_{j},j=1,\dots,n`
- カメラ姿勢 :math:`\bf{a}_{i} = [\bf{t}_{i}, \bf{\omega}_{i}],i=1,\dots,m`
  ただし :math:`\bf{t} \in \mathbb{R}^{3}` は並進を表すベクトルであり，:math:`\bf{\omega} \in \mathfrak{so}(3)` はカメラの回転を表す回転行列 :math:`R \in \mathbb{R}^{3 \times 3}` に対応するリー代数の元である．


入力
~~~~


各視点から観測されたランドマークの像の集合 :math:`\rm{X}`

.. math::
    \rm{X} = \begin{bmatrix}
        \bf{x}^{\top}_{11},
        \dots,
        \bf{x}^{\top}_{1m},
        \bf{x}^{\top}_{21},
        \dots,
        \bf{x}^{\top}_{2m},
        \dots,
        \bf{x}^{\top}_{n1},
        \dots,
        \bf{x}^{\top}_{nm}
    \end{bmatrix}


目的
----

投影モデルを :math:`\rm{Q}(\bf{a}_{i},\bf{b}_{j})` とし，以下の誤差関数を最小化するような :math:`\rm{P} = \left[\bf{a}, \bf{b}\right] = \left[ \bf{a}^{\top}_{1}, \dots, \bf{a}^{\top}_{m}, \bf{b}^{\top}_{1}, \dots, \bf{b}^{\top}_{n} \right]` を求める．

.. math::
    E(\rm{P}) = \begin{align}
    \sum_{i=1}^{n} \sum_{j=1}^{m} d_{\Cov_{\bf{x}_{ij}}}(\rm{Q}(\bf{a}_{j}, \bf{b}_{i}), \bf{x}_{ij})^{2}
    \end{align}


ここで :math:`d_{\Cov_{\bf{x}}}` は :math:`\bf{x}` に対応する分散共分散行列を :math:`\Cov_{\bf{x}}` として

.. math::
    d_{\Cov_{\bf{x}}}(\bf{x}_{1}, \bf{x}_{2}) =
    \sqrt{(\bf{x}_{1} - \bf{x}_{2})^{\top} \Cov^{-1}_{\bf{x}} (\bf{x}_{1} - \bf{x}_{2})}

で定義される距離関数である．

.. math::
    \hat{\rm{X}}
    = \begin{bmatrix}
        \hat{\bf{x}}^{\top}_{11},
        \dots,
        \hat{\bf{x}}^{\top}_{1m},
        \hat{\bf{x}}^{\top}_{21},
        \dots,
        \hat{\bf{x}}^{\top}_{2m},
        \dots,
        \hat{\bf{x}}^{\top}_{n1},
        \dots,
        \hat{\bf{x}}^{\top}_{nm}
    \end{bmatrix}^{\top} \\
    :label: definition-X

.. math::
    \hat{\bf{x}}_{ij}
    = \rm{Q}(\bf{a}_{j}, \bf{b}_{i})
    :label: definition-Q

.. math::
    \Cov_{\rm{X}}
    = diag(
        \Cov_{\bf{x}_{11}},
        \dots,
        \Cov_{\bf{x}_{1m}},
        \Cov_{\bf{x}_{21}},
        \dots,
        \Cov_{\bf{x}_{2m}},
        \dots,
        \Cov_{\bf{x}_{n1}},
        \dots,
        \Cov_{\bf{x}_{nm}}
    )
    :label: definition-sigma

とおけば，誤差を次のように表現することができる．

.. math::
    E(\rm{P})
    = (\rm{X}-\hat{\rm{X}})^{\top} \Cov_{\rm{X}}^{-1} (\rm{X}-\hat{\rm{X}})


解法の概要
----------

SBAでは，誤差関数を最小化するような :math:`\rm{P}` を見つけるため， :math:`\rm{P}^{(t)}` を逐次的に更新し，誤差関数を探索する．すなわち，時刻 :math:`t` における :math:`\rm{P}` の更新量を :math:`\D_{\rm{P}}^{(t)} = \left[ \D_{\bf{a}_{1}}^{\top}, \dots, \D_{\bf{a}_{m}}^{\top}, \D_{\bf{b}_{1}}^{\top}, \dots, \D_{\bf{b}_{n}}^{\top} \right]` ` として，

.. math::
    \rm{P}^{(t+1)} \leftarrow \rm{P}^{(t)} + \D_{\rm{P}}^{(t)}
    :label: parameter-update

というふうに :math:`\rm{P}^{(t)}` を更新することで誤差関数を最小化するような :math:`\rm{P}` を見つける．

更新量 :math:`\D_{\rm{P}}^{(t)}` の計算には LM法_ [#Levenberg_1944]_ を用いる．
更新量 :math:`\D_{\rm{P}}` は次の線型方程式を解くことによって得られる．

.. _LM法: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

.. math::
    \left[
        \rm{J}^{\top} \rm{\Cov}^{-1} \rm{J} + \lambda \rm{I}
    \right]
    \D_{\rm{P}}^{(t)}
    = \rm{J}^{\top} \rm{\Cov}^{-1} \left[ \rm{X} - \hat{\rm{X}} \right] \\
    :label: lm-update

:math:`\bf{J}` は :math:`\hat{\rm{X}}` のヤコビ行列 :math:`\rm{J} = \frac{\partial \hat{\rm{X}}}{\partial \rm{P}} \rvert_{\rm{P}=\rm{P}^{(t)}}` であり， :math:`\lambda \in \mathbb{R}, \lambda \geq 0` は damping parameter である．

SBAでは，:math:`\rm{J}` の構造に着目し， :eq:`lm-update` をより小さい複数の線型方程式に分解する．さらに，分解によって得られた方程式がスパースな行列によって構成されていることに着目し，計算を高速化している．


線型方程式の分解
~~~~~~~~~~~~~~~~

まず :math:`\rm{J}` を分解する． :math:`\rm{P}` の定義より， :math:`\rm{A} = \frac{\partial \hat{\rm{X}}}{\partial \bf{a}},\rm{B} = \frac{\partial \hat{\rm{X}}}{\partial \bf{b}}` とおけば， :math:`\rm{J}` は

.. math::
    \rm{J} = \frac{\partial \hat{\rm{X}}}{\partial \rm{P}}
    = \frac{\partial \hat{\rm{X}}}{\partial (\rm{a}, \rm{b})} = \left[ A, B \right]
    :label: decomposition-J

と書ける．

次に :eq:`lm-update` の右辺を分解する． :eq:`decomposition-J` を用いると， :eq:`lm-update` の右辺は

.. math::
    \begin{align}
        \bf{\epsilon}_{\bf{a}} &= A^{\top} \Cov^{-1} (\rm{X} - \hat{\rm{X}}) \\
        \bf{\epsilon}_{\bf{b}} &= B^{\top} \Cov^{-1} (\rm{X} - \hat{\rm{X}})
    \end{align}

とおくことによって，

.. math::
    \rm{J}^{\top} \rm{\Cov}^{-1} (\rm{X} - \hat{\rm{X}})
    = \begin{bmatrix} \bf{\epsilon}_{\bf{a}} \\ \bf{\epsilon}_{\bf{b}} \end{bmatrix}

と書ける．

さらに :eq:`lm-update` の左辺を分解する．
左辺の :math:`\rm{J}^{\top} \rm{\Cov}^{-1} \rm{J}` という項は大きく4つの行列に分解することができる．

.. math::
    \begin{align}
        \rm{J}^{\top} \rm{\Cov}^{-1} \rm{J}
        &= \begin{bmatrix}
            A^{\top} \\ B^{\top}
        \end{bmatrix}
        \Cov^{-1}
        \begin{bmatrix}
            A & B
        \end{bmatrix} \\
        &= \begin{bmatrix}
            A^{\top} \Cov^{-1} A & A^{\top} \Cov^{-1} B \\
            B^{\top} \Cov^{-1} A & B^{\top} \Cov^{-1} B
        \end{bmatrix} \\
        &= \begin{bmatrix}
            \rm{U} & \rm{W} \\
            \rm{W}^{\top} & \rm{V}
        \end{bmatrix}
    \end{align}
    :label: left-side-decomposition

以上の結果を用いると， :eq:`lm-update` は

.. math::
    \left[
    \begin{bmatrix}
        \rm{U} & \rm{W} \\
        \rm{W}^{\top} & \rm{V}
    \end{bmatrix}
    +
    \begin{bmatrix}
        \lambda \rm{I} & \rm{0} \\
        \rm{0} & \lambda \rm{I}
    \end{bmatrix}
    \right]
    \begin{bmatrix}
        \Da \\
        \Db
    \end{bmatrix}
    =
    \begin{bmatrix}
        \bf{\epsilon}_{\bf{a}} \\
        \bf{\epsilon}_{\bf{b}}
    \end{bmatrix}

という形にすることができる．
さらに，

.. math::
    \begin{align}
        \rm{U}^{*} &= \rm{U} + \lambda \rm{I} \\
        \VStar &= \rm{V} + \lambda \rm{I}
    \end{align}

とおけば，

.. math::
    \begin{bmatrix}
        \rm{U}^{*} & \rm{W} \\
        \rm{W}^{\top} & \VStar
    \end{bmatrix}
    \begin{bmatrix}
        \Da \\
        \Db
    \end{bmatrix}
    =
    \begin{bmatrix}
        \bf{\epsilon}_{\bf{a}} \\
        \bf{\epsilon}_{\bf{b}}
    \end{bmatrix}

となる．この両辺に

.. math::
    \begin{bmatrix}
        \rm{I} & -\rm{W}{\VStar}^{-1} \\
        \rm{0} & \rm{I}
    \end{bmatrix}

という行列を左から作用させると，

.. math::
    \begin{bmatrix}
        \rm{I} & -\rm{W}{\VStar}^{-1} \\
        \rm{0} & \rm{I}
    \end{bmatrix}
    \begin{bmatrix}
        \rm{U}^{*} & \rm{W} \\
        \rm{W}^{\top} & \VStar
    \end{bmatrix}
    \begin{bmatrix}
        \Da \\
        \Db
    \end{bmatrix}
    =
    \begin{bmatrix}
        \rm{I} & -\rm{W}{\VStar}^{-1} \\
        \rm{0} & \rm{I}
    \end{bmatrix}
    \begin{bmatrix}
        \bf{\epsilon}_{\bf{a}} \\
        \bf{\epsilon}_{\bf{b}}
    \end{bmatrix} \\
    :label: left-multiplication

.. math::
    \begin{bmatrix}
        \rm{U}^{*} - \rm{W}{\VStar}^{-1}\rm{W}^{\top} & \rm{0} \\
        \rm{W}^{\top} & \VStar
    \end{bmatrix}
    \begin{bmatrix}
        \Da \\
        \Db
    \end{bmatrix}
    =
    \begin{bmatrix}
        \bf{\epsilon}_{\bf{a}} - \rm{W}{\VStar}^{-1}\bf{\epsilon}_{\bf{b}} \\
        \bf{\epsilon}_{\bf{b}}
    \end{bmatrix}
    :label: affected-from-left

という形にすることができる．ここから2つの方程式を取り出す．
すると， :eq:`affected-from-left` において左辺の行列の右上が :math:`\rm{0}` になったことから， :math:`\Db` を含まない :math:`\Da` についての式 :eq:`derivation-da` を得ることができる．

.. math::
    (\rm{U}^{*} - \rm{W}{\VStar}^{-1}\rm{W}^{\top}) \Da
    = \bf{\epsilon}_{\bf{a}} - \rm{W}{\VStar}^{-1}\bf{\epsilon}_{\bf{b}}
    :label: derivation-da

.. math::
    \VStar \Db
    = \bf{\epsilon}_{\bf{b}} - \rm{W}^{\top} \Da
    :label: derivation-db

したがって，:eq:`derivation-da` を先に解き，得られた :math:`\Da` を :eq:`derivation-db` に代入すれば :math:`\Db` を得ることができる．


具体的な計算
------------

前節では，LM法を分解し，より少ない計算量で更新量 :math:`\bf{\D}_{\rm{P}}` を求める方法を述べた．
ここでは，実際にヤコビ行列 :math:`\rm{J}` を計算し，その具体的なかたちを求める．

まず，ヤコビ行列 :math:`\rm{J}` はスパースな行列になる．

これは，:math:`\forall j \neq k` について

.. math::
    \frac{\partial \rm{Q}(\bf{a}_{j}, \bf{b}_{i})}{\partial \bf{a}_{k}} = \bf{0}

:math:`\forall i \neq k` について

.. math::
    \frac{\partial \rm{Q}(\bf{a}_{j}, \bf{b}_{i})}{\partial \bf{b}_{k}} = \bf{0}

が成り立つためである．


例えば，:math:`n=4` ，:math:`m=3` のとき，
:math:`\rm{A}_{ij}=\frac{\partial \rm{Q}(\bf{a}_{j}, \bf{b}_{i})}{\partial \bf{a}_{j}}` ，
:math:`\rm{B}_{ij}=\frac{\partial \rm{Q}(\bf{a}_{j}, \bf{b}_{i})}{\partial \bf{b}_{i}}`
とおけば，:math:`\rm{J}` は

.. math::
    \rm{J} = \begin{bmatrix}
        \rm{A}_{11} &      \bf{0} &      \bf{0} & \rm{B}_{11} &      \bf{0} &      \bf{0} &      \bf{0} \\
        \bf{0}      & \rm{A}_{12} &      \bf{0} & \rm{B}_{12} &      \bf{0} &      \bf{0} &      \bf{0} \\
        \bf{0}      &      \bf{0} & \rm{A}_{13} & \rm{B}_{13} &      \bf{0} &      \bf{0} &      \bf{0} \\
        \rm{A}_{21} &      \bf{0} &      \bf{0} &      \bf{0} & \rm{B}_{21} &      \bf{0} &      \bf{0} \\
        \bf{0}      & \rm{A}_{22} &      \bf{0} &      \bf{0} & \rm{B}_{22} &      \bf{0} &      \bf{0} \\
        \bf{0}      &      \bf{0} & \rm{A}_{23} &      \bf{0} & \rm{B}_{23} &      \bf{0} &      \bf{0} \\
        \rm{A}_{31} &      \bf{0} &      \bf{0} &      \bf{0} &      \bf{0} & \rm{B}_{31} &      \bf{0} \\
        \bf{0}      & \rm{A}_{32} &      \bf{0} &      \bf{0} &      \bf{0} & \rm{B}_{32} &      \bf{0} \\
        \bf{0}      &      \bf{0} & \rm{A}_{33} &      \bf{0} &      \bf{0} & \rm{B}_{33} &      \bf{0} \\
        \rm{A}_{41} &      \bf{0} &      \bf{0} &      \bf{0} &      \bf{0} &      \bf{0} & \rm{B}_{41} \\
        \bf{0}      & \rm{A}_{42} &      \bf{0} &      \bf{0} &      \bf{0} &      \bf{0} & \rm{B}_{42} \\
        \bf{0}      &      \bf{0} & \rm{A}_{43} &      \bf{0} &      \bf{0} &      \bf{0} & \rm{B}_{43} \\
    \end{bmatrix}
    :label: concrete-form-J

となる．

では :math:`\rm{A}_{ij}` や :math:`\rm{B}_{ij}` の具体的なかたちを求めてみよう．

姿勢パラメータに関する微分
~~~~~~~~~~~~~~~~~~~~~~~~~~


姿勢パラメータ :math:`\bf{a} = \left[ \bf{t}, \bf{\omega} \right]` に関する微分 :math:`\rm{B}=\frac{\partial \rm{Q}(\bf{a}, \bf{b})}{\partial \bf{b}}` は次のようになる．


.. math::
    \begin{align}
    \frac{\partial \hat{\bf{x}}}{\partial \bf{t}}
    &= \frac{\partial \pi(\bf{p})}{\partial \bf{p}}
       \bigg\rvert_{\bf{p}=\rm{K}(\rm{R}\bf{b} + \bf{t})}
       \cdot
       \rm{K}
       \cdot
       \frac{\partial (\rm{R}(\bf{\omega})\bf{b} + \bf{v})}{\partial \bf{v}}
       \bigg\rvert_{\bf{v}=\bf{t}} \\
    &= \frac{\partial \pi(\bf{p})}{\partial \bf{p}}
       \bigg\rvert_{\bf{p}=\rm{K}(\rm{R}\bf{b} + \bf{t})}
       \cdot
       \rm{K}
    \end{align}


.. math::
    \begin{align}
    \frac{\partial \hat{\bf{x}}}{\partial \bf{\omega}}
    &= \frac{\partial \pi(\bf{p})}{\partial \bf{p}}
       \bigg\rvert_{\bf{p}=\rm{K}(\rm{R}\bf{b} + \bf{t})}
       \cdot
       \rm{K}
       \cdot
       \frac{\partial (\rm{R}(\bf{v})\bf{b} + \bf{t})}{\partial \bf{v}}
       \bigg\rvert_{\bf{v}=\bf{\omega}} \\
    &= \frac{\partial \pi(\bf{p})}{\partial \bf{p}}
       \bigg\rvert_{\bf{p}=\rm{K}(\rm{R}\bf{b} + \bf{t})}
       \cdot
       \rm{K}
       \cdot
       \frac{\partial (\rm{R}(\bf{v})\bf{b})}{\partial \bf{v}}
       \bigg\rvert_{\bf{v}=\bf{\omega}}
    \end{align}


ここで， :math:`\frac{\partial (\rm{R}(\bf{v})\bf{b})}{\partial \bf{v}}` は [#Gallego_et_al_2015]_ による計算結果を用いることができる

.. math::
   \frac{\partial (\rm{R}(\bf{v})\bf{b})}{\partial \bf{v}}
   = -\rm{R}(\bf{v}) \left[ \bf{b} \right]_{\times}
     \frac{
        \bf{v}\bf{v}^{\top} +
        (\rm{R}(\bf{v})^{\top} - \rm{I}) \left[ \bf{v} \right]_{\times}
     }{||\bf{v}||^{2}}


3次元点座標に関する微分
~~~~~~~~~~~~~~~~~~~~~~~

3次元点の座標 :math:`\bf{b}` に関する微分 :math:`\rm{B}=\frac{\partial \rm{Q}(\bf{a}, \bf{b})}{\partial \bf{b}}` は次のようになる．

.. math::
    \begin{align}
    \frac{\partial \hat{\bf{x}}}{\partial \bf{b}}
    &= \frac{\partial \pi(\bf{p})}{\partial \bf{p}}
       \bigg\rvert_{\bf{p}=\rm{K}(\rm{R}\bf{b} + \bf{t})}
       \cdot
       \rm{K}
       \cdot
       \frac{\partial (\rm{R}(\bf{\omega})\bf{v} + \bf{t})}{\partial \bf{v}}
       \bigg\rvert_{\bf{v}=\bf{b}} \\
    &= \frac{\partial \pi(\bf{p})}{\partial \bf{p}}
       \bigg\rvert_{\bf{p}=\rm{K}(\rm{R}\bf{b} + \bf{t})}
       \cdot
       \rm{K}
       \cdot
       \rm{R}(\bf{\omega})
    \end{align}


以上より， :math:`\rm{A}_{ij}` と :math:`\rm{B}_{ij}` の具体的なかたちを求めることができた．あとは，

    1. 上記で得られた :math:`\rm{A}_{ij}` と :math:`\rm{B}_{ij}` :eq:`concrete-form-J` に代入して :math:`\rm{J}` を求める
    2. :eq:`left-side-decomposition` にしたがって :math:`\rm{U},\rm{V},\rm{W}` を求める
    3. :eq:`derivation-da` と :eq:`derivation-db` によって姿勢パラメータ :math:`\bf{a}` と3次元点の座標 :math:`\bf{b}` それぞれについての更新量 :math:`\Da` と :math:`\Db` を求める

という3つのステップによって更新量を求めることができる．


計算量の削減
~~~~~~~~~~~~

前節までで更新量の計算 :eq:`lm-update` を2つの計算 :eq:`derivation-da` :eq:`derivation-db` に分解する過程を見た．SBAは， :math:`\VStar` がスパースであるという性質に基づいて計算量を削減している．


:eq:`concrete-form-J` で定義された :math:`\rm{J}` を用いて :math:`\VStar` を計算すると次のようになる．


.. math::
    \VStar = \begin{bmatrix}
        \VStar_{1} & \rm{0} & \rm{0} & \rm{0} \\
        \rm{0} & \VStar_{2} & \rm{0} & \rm{0} \\
        \rm{0} & \rm{0} & \VStar_{3} & \rm{0} \\
        \rm{0} & \rm{0} & \rm{0} & \VStar_{4} \\
    \end{bmatrix}

ただし

.. math::
    \begin{align}
        \rm{V}_{i}
        &= \sum_{j=1}^{m} \rm{B}_{ij}^{\top} \Cov_{ij}^{-1} \rm{B}_{ij} \\
        \VStar_{i}
        &= \rm{V}_{i} + \lambda \rm{I}.
    \end{align}


| :eq:`derivation-da` には :math:`{\VStar}` の逆行列が両辺に含まれている．また， :eq:`derivation-db` を解いて :math:`\Db` を得る際にも両辺に左から :math:`{\VStar}` の逆行列をかける必要がある．
| 一般的な :math:`n \times n` 行列の逆行列の計算は :math:`O(n^{2.3})` 〜 :math:`O(n^{3})` 程度のオーダーとなってしまうため， :math:`\VStar` のサイズが大きいとその逆行列を求めるのに多大なコストがかかってしまう．しかし， :math:`\VStar` がスパースな行列であることに着目すると， :math:`\VStar` の逆行列は

.. math::
    {\VStar}^{-1} = \begin{bmatrix}
        {\VStar}^{-1}_{1} & \rm{0} & \rm{0} & \rm{0} \\
        \rm{0} & {\VStar}^{-1}_{2} & \rm{0} & \rm{0} \\
        \rm{0} & \rm{0} & {\VStar}^{-1}_{3} & \rm{0} \\
        \rm{0} & \rm{0} & \rm{0} & {\VStar}^{-1}_{4} \\
    \end{bmatrix}

となるため， :math:`\VStar_{i},i=1,\dots,m` のそれぞれについて逆行列を求めればよいことがわかる．したがって :math:`\VStar` の逆行列の計算量は視点数 :math:`m` に対して線型に増加することになり， :math:`\VStar` の逆行列を直接求めるのと比較すると計算量を一気に削減できる．

:math:`\Da` を求める際には， :math:`\rm{S} = \rm{U}^{*} - \rm{W}{\VStar}^{-1}\rm{W}^{\top}` の逆行列を :eq:`derivation-da` の両辺に左からかける必要がある．しかし，一般的にランドマーク数 :math:`n` よりもカメラの視点数 :math:`m` の方が圧倒的に小さい :math:`(m \ll n)` ため， :math:`\rm{S}` のサイズは :math:`\VStar` と比べると圧倒的に小さい．したがって， :math:`\rm{S}` の逆行列を求める処理は全体の計算量にはほとんど影響しない．

問題のサイズ(視点数や復元対象となるランドマークの数)が大きいときは， :eq:`lm-update` を直接解いて :math:`\bf{\D}_{\rm{P}}` を得るよりも， :eq:`derivation-da` と :eq:`derivation-db` によって :math:`\Da` と :math:`\Db` をそれぞれ計算し結合することで :math:`\bf{\D}_{\rm{P}}` を得た方が圧倒的に高速である．

| :eq:`lm-update` ， :eq:`derivation-da` ， :eq:`derivation-db` はいずれも線型方程式 :math:`\bf{y} = \rm{A}\bf{x},\; \bf{x} \in \mathbb{R}^{n}, \bf{y} \in \mathbb{R}^{m}, \rm{A} \in \mathbb{R}^{n \times m}` のかたちをしているため，:eq:`lm-update` から直接 :math:`\bf{\D}_{\rm{P}}` を得る場合と， :eq:`derivation-da` ， :eq:`derivation-db` をそれぞれ解いて :math:`\bf{\D}_{\rm{P}}` を得る場合のどちらも線型方程式を解くことになる．
| 線型方程式の解は :math:`\bf{x} = (\rm{A}^{\top}\rm{A})^{-1}\rm{A}^{\top}\bf{y}` を解くことで得られるが，
  すなわち，問題のサイズが大きくなると計算量が急激に増加するため，大きな問題を直接解くよりも，大きな問題を複数の小さな問題に分割して解いた方が計算コストを抑えることができる．
| SBAでは，式 :eq:`lm-update` を直接解く代わりに，それを小さく分割して得た :eq:`derivation-da` と :eq:`derivation-db` をそれぞれ解くことによって，計算コストを削減している．



.. [#Gallego_et_al_2015] Gallego, Guillermo, and Anthony Yezzi. "A compact formula for the derivative of a 3-D rotation in exponential coordinates." Journal of Mathematical Imaging and Vision 51.3 (2015): 378-384.
.. [#Levenberg_1944] Levenberg, Kenneth. "A method for the solution of certain non-linear problems in least squares." Quarterly of applied mathematics 2.2 (1944): 164-168.
