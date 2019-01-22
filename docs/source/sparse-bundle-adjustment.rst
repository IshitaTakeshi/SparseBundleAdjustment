========================
Spamse Bundle Adjustment
========================

.. math::
    \def\mbf#1{{\mathbf #1}}
    \def\D{{\delta}}
    \def\Cov{{\Sigma}}
    \def\VStar{{V^{*}}}
    \def\Db{{\mbf{\D}_{\mbf{b}}}}
    \def\Da{{\mbf{\D}_{\mbf{a}}}}
    \def\DP{{\mbf{\D}_{P}}}

概要
====

Sparse Bundle Adjustment(SBA) [#Lourakis_et_al_2015]_ はSfMの手法の一種である．
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
========

入力
----


各視点から観測されたランドマークの像の集合 :math:`X`

.. math::
    X = \begin{bmatrix}
        \mbf{x}^{\top}_{11},
        \dots,
        \mbf{x}^{\top}_{1m},
        \mbf{x}^{\top}_{21},
        \dots,
        \mbf{x}^{\top}_{2m},
        \dots,
        \mbf{x}^{\top}_{n1},
        \dots,
        \mbf{x}^{\top}_{nm}
    \end{bmatrix}


出力
----

- 3次元空間におけるランドマーク座標 :math:`\mbf{b}_{j},j=1,\dots,n`
- カメラ姿勢 :math:`\mbf{a}_{i} = [\mbf{t}_{i}, \mbf{\omega}_{i}],i=1,\dots,m`
  ただし :math:`\mbf{t} \in \mathbb{R}^{3}` は並進を表すベクトルであり，:math:`\mbf{\omega} \in \mathfrak{so}(3)` はカメラの回転を表す回転行列 :math:`R \in \mathbb{R}^{3 \times 3}` に対応するリー代数の元である．


目的
----


投影モデルを :math:`Q(\mbf{a}_{i},\mbf{b}_{j})` とし，以下の誤差関数を最小化するような :math:`P = \left[\mbf{a}, \mbf{b}\right] = \left[ \mbf{a}^{\top}_{1}, \dots, \mbf{a}^{\top}_{m}, \mbf{b}^{\top}_{1}, \dots, \mbf{b}^{\top}_{n} \right]` を求める．

.. math::
    E(P) = \begin{align}
    \sum_{i=1}^{n} \sum_{j=1}^{m} d_{\Cov_{\mbf{x}_{ij}}}(Q(\mbf{a}_{j}, \mbf{b}_{i}), \mbf{x}_{ij})^{2}
    \end{align}
    :label: error-function


ここで :math:`d_{\Cov_{\mbf{x}}}` は :math:`\mbf{x}` に対応する分散共分散行列を :math:`\Cov_{\mbf{x}}` として

.. math::
    d_{\Cov_{\mbf{x}}}(\mbf{x}_{1}, \mbf{x}_{2}) =
    \sqrt{(\mbf{x}_{1} - \mbf{x}_{2})^{\top} \Cov^{-1}_{\mbf{x}} (\mbf{x}_{1} - \mbf{x}_{2})}

で定義される距離関数である．

.. math::
    \hat{X}
    = \begin{bmatrix}
        \hat{\mbf{x}}^{\top}_{11},
        \dots,
        \hat{\mbf{x}}^{\top}_{1m},
        \hat{\mbf{x}}^{\top}_{21},
        \dots,
        \hat{\mbf{x}}^{\top}_{2m},
        \dots,
        \hat{\mbf{x}}^{\top}_{n1},
        \dots,
        \hat{\mbf{x}}^{\top}_{nm}
    \end{bmatrix}^{\top} \\
    :label: definition-X

.. math::
    \hat{\mbf{x}}_{ij}
    = Q(\mbf{a}_{j}, \mbf{b}_{i})
    :label: definition-Q

.. math::
    \Cov_{X}
    = diag(
        \Cov_{\mbf{x}_{11}},
        \dots,
        \Cov_{\mbf{x}_{1m}},
        \Cov_{\mbf{x}_{21}},
        \dots,
        \Cov_{\mbf{x}_{2m}},
        \dots,
        \Cov_{\mbf{x}_{n1}},
        \dots,
        \Cov_{\mbf{x}_{nm}}
    )
    :label: definition-sigma

とおけば，誤差を次のように表現することができる．

.. math::
    E(P)
    = (X-\hat{X})^{\top} \Cov_{X}^{-1} (X-\hat{X})


解法の概要
==========

SBAでは，誤差関数を最小化するような :math:`P` を見つけるため， :math:`P^{(t)}` を逐次的に更新し，誤差関数を探索する．すなわち，時刻 :math:`t` における :math:`P` の更新量を :math:`\D_{P}^{(t)} = \left[ \D_{\mbf{a}_{1}}^{\top}, \dots, \D_{\mbf{a}_{m}}^{\top}, \D_{\mbf{b}_{1}}^{\top}, \dots, \D_{\mbf{b}_{n}}^{\top} \right]`  として，

.. math::
    P^{(t+1)} \leftarrow P^{(t)} + \D_{P}^{(t)}
    :label: parameter-update

というふうに :math:`P^{(t)}` を更新することで誤差関数を最小化するような :math:`P` を見つける．

更新量 :math:`\D_{P}^{(t)}` の計算には LM法_ [#Levenberg_1944]_ を用いる．
更新量 :math:`\D_{P}` は次の線型方程式を解くことによって得られる．

.. _LM法: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

.. math::
    \left[
        J^{\top} \Cov^{-1} J + \lambda I
    \right]
    \D_{P}^{(t)}
    = J^{\top} \Cov^{-1} \left[ X - \hat{X} \right] \\
    :label: lm-update

:math:`\mbf{J}` は :math:`\hat{X}` のヤコビ行列 :math:`J = \frac{\partial \hat{X}}{\partial P} \rvert_{P=P^{(t)}}` であり， :math:`\lambda \in \mathbb{R}, \lambda \geq 0` は damping parameter である．

SBAでは，:math:`J` の構造に着目し， :eq:`lm-update` をより小さい複数の線型方程式に分解する．さらに，分解によって得られた方程式がスパースな行列によって構成されていることに着目し，計算を高速化している．

解法
====

線型方程式の分解
----------------

まず :math:`J` を分解する． :math:`P` の定義より， :math:`A = \frac{\partial \hat{X}}{\partial \mbf{a}},B = \frac{\partial \hat{X}}{\partial \mbf{b}}` とおけば， :math:`J` は

.. math::
    J = \frac{\partial \hat{X}}{\partial P}
    = \frac{\partial \hat{X}}{\partial (a, b)} = \left[ A, B \right]
    :label: decomposition-J

と書ける．

次に :eq:`lm-update` の右辺を分解する． :eq:`decomposition-J` を用いると， :eq:`lm-update` の右辺は

.. math::
    \begin{align}
        \mbf{\epsilon}_{\mbf{a}} &= A^{\top} \Cov^{-1} (X - \hat{X}) \\
        \mbf{\epsilon}_{\mbf{b}} &= B^{\top} \Cov^{-1} (X - \hat{X})
    \end{align}

とおくことによって，

.. math::
    J^{\top} \Cov^{-1} (X - \hat{X})
    = \begin{bmatrix} \mbf{\epsilon}_{\mbf{a}} \\ \mbf{\epsilon}_{\mbf{b}} \end{bmatrix}

と書ける．

さらに :eq:`lm-update` の左辺を分解する．
左辺の :math:`J^{\top} \Cov^{-1} J` という項は大きく4つの行列に分解することができる．

.. math::
    \begin{align}
        J^{\top} \Cov^{-1} J
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
            U & W \\
            W^{\top} & V
        \end{bmatrix}
    \end{align}
    :label: left-side-decomposition


以上の結果を用いると， :eq:`lm-update` は


.. math::
    \left[
    \begin{bmatrix}
        U & W \\
        W^{\top} & V
    \end{bmatrix}
    +
    \begin{bmatrix}
        \lambda I & 0 \\
        0 & \lambda I
    \end{bmatrix}
    \right]
    \begin{bmatrix}
        \Da \\
        \Db
    \end{bmatrix}
    =
    \begin{bmatrix}
        \mbf{\epsilon}_{\mbf{a}} \\
        \mbf{\epsilon}_{\mbf{b}}
    \end{bmatrix}

という形にすることができる．
さらに，

.. math::
    \begin{align}
        U^{*} &= U + \lambda I \\
        \VStar &= V + \lambda I
    \end{align}

とおけば，

.. math::
    \begin{bmatrix}
        U^{*} & W \\
        W^{\top} & \VStar
    \end{bmatrix}
    \begin{bmatrix}
        \Da \\
        \Db
    \end{bmatrix}
    =
    \begin{bmatrix}
        \mbf{\epsilon}_{\mbf{a}} \\
        \mbf{\epsilon}_{\mbf{b}}
    \end{bmatrix}

となる．この両辺に

.. math::
    \begin{bmatrix}
        I & -W{\VStar}^{-1} \\
        0 & I
    \end{bmatrix}

という行列を左から作用させると，

.. math::
    \begin{bmatrix}
        I & -W{\VStar}^{-1} \\
        0 & I
    \end{bmatrix}
    \begin{bmatrix}
        U^{*} & W \\
        W^{\top} & \VStar
    \end{bmatrix}
    \begin{bmatrix}
        \Da \\
        \Db
    \end{bmatrix}
    =
    \begin{bmatrix}
        I & -W{\VStar}^{-1} \\
        0 & I
    \end{bmatrix}
    \begin{bmatrix}
        \mbf{\epsilon}_{\mbf{a}} \\
        \mbf{\epsilon}_{\mbf{b}}
    \end{bmatrix} \\
    :label: left-multiplication

.. math::
    \begin{bmatrix}
        U^{*} - W{\VStar}^{-1}W^{\top} & 0 \\
        W^{\top} & \VStar
    \end{bmatrix}
    \begin{bmatrix}
        \Da \\
        \Db
    \end{bmatrix}
    =
    \begin{bmatrix}
        \mbf{\epsilon}_{\mbf{a}} - W{\VStar}^{-1}\mbf{\epsilon}_{\mbf{b}} \\
        \mbf{\epsilon}_{\mbf{b}}
    \end{bmatrix}
    :label: affected-from-left

という形にすることができる．ここから2つの方程式を取り出す．
すると， :eq:`affected-from-left` において左辺の行列の右上が :math:`0` になったことから， :math:`\Db` を含まない :math:`\Da` についての式 :eq:`derivation-da` を得ることができる．

.. math::
    (U^{*} - W{\VStar}^{-1}W^{\top}) \Da
    = \mbf{\epsilon}_{\mbf{a}} - W{\VStar}^{-1}\mbf{\epsilon}_{\mbf{b}}
    :label: derivation-da

.. math::
    \VStar \Db
    = \mbf{\epsilon}_{\mbf{b}} - W^{\top} \Da
    :label: derivation-db

したがって，:eq:`derivation-da` を先に解き，得られた :math:`\Da` を :eq:`derivation-db` に代入すれば :math:`\Db` を得ることができる．


具体的な計算
------------

前節では，LM法を分解し，より少ない計算量で更新量 :math:`\DP` を求める方法を述べた．
ここでは，実際にヤコビ行列 :math:`J` を計算し，その具体的なかたちを求める．

まず，ヤコビ行列 :math:`J` はスパースな行列になる．

これは，:math:`\forall j \neq k` について

.. math::
    \frac{\partial Q(\mbf{a}_{j}, \mbf{b}_{i})}{\partial \mbf{a}_{k}} = \mbf{0}

:math:`\forall i \neq k` について

.. math::
    \frac{\partial Q(\mbf{a}_{j}, \mbf{b}_{i})}{\partial \mbf{b}_{k}} = \mbf{0}

が成り立つためである．


例えば，:math:`n=4` ，:math:`m=3` のとき，
:math:`A_{ij}=\frac{\partial Q(\mbf{a}_{j}, \mbf{b}_{i})}{\partial \mbf{a}_{j}}` ，
:math:`B_{ij}=\frac{\partial Q(\mbf{a}_{j}, \mbf{b}_{i})}{\partial \mbf{b}_{i}}`
とおけば，:math:`J` は

.. math::
    J = \begin{bmatrix}
        A_{11} &      \mbf{0} &      \mbf{0} & B_{11} &      \mbf{0} &      \mbf{0} &      \mbf{0} \\
        \mbf{0}      & A_{12} &      \mbf{0} & B_{12} &      \mbf{0} &      \mbf{0} &      \mbf{0} \\
        \mbf{0}      &      \mbf{0} & A_{13} & B_{13} &      \mbf{0} &      \mbf{0} &      \mbf{0} \\
        A_{21} &      \mbf{0} &      \mbf{0} &      \mbf{0} & B_{21} &      \mbf{0} &      \mbf{0} \\
        \mbf{0}      & A_{22} &      \mbf{0} &      \mbf{0} & B_{22} &      \mbf{0} &      \mbf{0} \\
        \mbf{0}      &      \mbf{0} & A_{23} &      \mbf{0} & B_{23} &      \mbf{0} &      \mbf{0} \\
        A_{31} &      \mbf{0} &      \mbf{0} &      \mbf{0} &      \mbf{0} & B_{31} &      \mbf{0} \\
        \mbf{0}      & A_{32} &      \mbf{0} &      \mbf{0} &      \mbf{0} & B_{32} &      \mbf{0} \\
        \mbf{0}      &      \mbf{0} & A_{33} &      \mbf{0} &      \mbf{0} & B_{33} &      \mbf{0} \\
        A_{41} &      \mbf{0} &      \mbf{0} &      \mbf{0} &      \mbf{0} &      \mbf{0} & B_{41} \\
        \mbf{0}      & A_{42} &      \mbf{0} &      \mbf{0} &      \mbf{0} &      \mbf{0} & B_{42} \\
        \mbf{0}      &      \mbf{0} & A_{43} &      \mbf{0} &      \mbf{0} &      \mbf{0} & B_{43} \\
    \end{bmatrix}
    :label: concrete-form-J

となる．

では :math:`A_{ij}` や :math:`B_{ij}` の具体的なかたちを求めてみよう．


姿勢パラメータに関する微分
~~~~~~~~~~~~~~~~~~~~~~~~~~


姿勢パラメータ :math:`\mbf{a} = \left[ \mbf{t}, \mbf{\omega} \right]` に関する微分 :math:`B=\frac{\partial Q(\mbf{a}, \mbf{b})}{\partial \mbf{b}}` は次のようになる．


.. math::
    \begin{align}
    \frac{\partial \hat{\mbf{x}}}{\partial \mbf{t}}
    &= \frac{\partial \pi(\mbf{p})}{\partial \mbf{p}}
       \bigg\rvert_{\mbf{p}=K(R\mbf{b} + \mbf{t})}
       \cdot
       K
       \cdot
       \frac{\partial (R(\mbf{\omega})\mbf{b} + \mbf{v})}{\partial \mbf{v}}
       \bigg\rvert_{\mbf{v}=\mbf{t}} \\
    &= \frac{\partial \pi(\mbf{p})}{\partial \mbf{p}}
       \bigg\rvert_{\mbf{p}=K(R\mbf{b} + \mbf{t})}
       \cdot
       K
    \end{align}


.. math::
    \begin{align}
    \frac{\partial \hat{\mbf{x}}}{\partial \mbf{\omega}}
    &= \frac{\partial \pi(\mbf{p})}{\partial \mbf{p}}
       \bigg\rvert_{\mbf{p}=K(R\mbf{b} + \mbf{t})}
       \cdot
       K
       \cdot
       \frac{\partial (R(\mbf{v})\mbf{b} + \mbf{t})}{\partial \mbf{v}}
       \bigg\rvert_{\mbf{v}=\mbf{\omega}} \\
    &= \frac{\partial \pi(\mbf{p})}{\partial \mbf{p}}
       \bigg\rvert_{\mbf{p}=K(R\mbf{b} + \mbf{t})}
       \cdot
       K
       \cdot
       \frac{\partial (R(\mbf{v})\mbf{b})}{\partial \mbf{v}}
       \bigg\rvert_{\mbf{v}=\mbf{\omega}}
    \end{align}


ここで， :math:`\frac{\partial (R(\mbf{v})\mbf{b})}{\partial \mbf{v}}` は [#Gallego_et_al_2015]_ による計算結果を用いることができる

.. math::
   \frac{\partial (R(\mbf{v})\mbf{b})}{\partial \mbf{v}}
   = -R(\mbf{v}) \left[ \mbf{b} \right]_{\times}
     \frac{
        \mbf{v}\mbf{v}^{\top} +
        (R(\mbf{v})^{\top} - I) \left[ \mbf{v} \right]_{\times}
     }{||\mbf{v}||^{2}}


3次元点座標に関する微分
~~~~~~~~~~~~~~~~~~~~~~~

3次元点の座標 :math:`\mbf{b}` に関する微分 :math:`B=\frac{\partial Q(\mbf{a}, \mbf{b})}{\partial \mbf{b}}` は次のようになる．

.. math::
    \begin{align}
    \frac{\partial \hat{\mbf{x}}}{\partial \mbf{b}}
    &= \frac{\partial \pi(\mbf{p})}{\partial \mbf{p}}
       \bigg\rvert_{\mbf{p}=K(R\mbf{b} + \mbf{t})}
       \cdot
       K
       \cdot
       \frac{\partial (R(\mbf{\omega})\mbf{v} + \mbf{t})}{\partial \mbf{v}}
       \bigg\rvert_{\mbf{v}=\mbf{b}} \\
    &= \frac{\partial \pi(\mbf{p})}{\partial \mbf{p}}
       \bigg\rvert_{\mbf{p}=K(R\mbf{b} + \mbf{t})}
       \cdot
       K
       \cdot
       R(\mbf{\omega})
    \end{align}


以上より， :math:`A_{ij}` と :math:`B_{ij}` の具体的なかたちを求めることができた．あとは，

    1. 上記で得られた :math:`A_{ij}` と :math:`B_{ij}` :eq:`concrete-form-J` に代入して :math:`J` を求める
    2. :eq:`left-side-decomposition` にしたがって :math:`U,V,W` を求める
    3. :eq:`derivation-da` と :eq:`derivation-db` によって姿勢パラメータ :math:`\mbf{a}` と3次元点の座標 :math:`\mbf{b}` それぞれについての更新量 :math:`\Da` と :math:`\Db` を求める

という3つのステップによって更新量を求めることができる．


計算量の削減
~~~~~~~~~~~~

前節までで更新量の計算 :eq:`lm-update` を2つの計算 :eq:`derivation-da` :eq:`derivation-db` に分解する過程を見た．

| :eq:`lm-update` ， :eq:`derivation-da` ， :eq:`derivation-db` はいずれも線型方程式とみなすことができる．
| 線型方程式 :math:`\mbf{y} = A\mbf{x},\; \mbf{x} \in \mathbb{R}^{n}, \mbf{y} \in \mathbb{R}^{m}, A \in \mathbb{R}^{n \times m}` の解は

.. math::
    \begin{align}
        \mbf{x}
        &= (A^{\top}A)^{-1}A^{\top}\mbf{y} \\
        &= K^{-1}A^{\top}\mbf{y} \\
        K &= A^{\top}A,
        K \in \mathbb{R}^{n \times n}
    \end{align}

| によって得られるが，行列 :math:`K` のサイズが大きくなると解を求めるための計算量が急激に増加する．これは， :math:`n \times n` 行列の逆行列を計算するアルゴリズムが :math:`O(n^{2.3})` 〜 :math:`O(n^{3})` 程度の計算量をもつことに起因する [#Coppersmith_et_al_1990]_ ．したがって，線型方程式を高速に解くには，問題の構造を見極め， :math:`K` の逆行列を直接計算することを避けて計算量を減らす必要がある．
| SBAでは， :eq:`lm-update` を直接解くのではなく，それを分割して得た :eq:`derivation-da` と :eq:`derivation-db` をそれぞれ解くことで :math:`\DP` を得ている．さらに， :math:`\VStar` がスパースであるという性質に基づいて計算量を大幅に削減している．



:eq:`concrete-form-J` で定義された :math:`J` を用いて :math:`\VStar` を計算すると次のようになる．


.. math::
    \VStar = \begin{bmatrix}
        \VStar_{1} & 0 & 0 & 0 \\
        0 & \VStar_{2} & 0 & 0 \\
        0 & 0 & \VStar_{3} & 0 \\
        0 & 0 & 0 & \VStar_{4} \\
    \end{bmatrix}

ただし

.. math::
    \begin{align}
        V_{i}
        &= \sum_{j=1}^{m} B_{ij}^{\top} \Cov_{ij}^{-1} B_{ij} \\
        \VStar_{i}
        &= V_{i} + \lambda I.
    \end{align}


| :eq:`derivation-da` には :math:`{\VStar}` の逆行列が両辺に含まれている．また， :eq:`derivation-db` を解いて :math:`\Db` を得る際にも両辺に左から :math:`{\VStar}` の逆行列をかける必要がある．
| :math:`\VStar` のサイズが大きいとその逆行列を求めるのに多大なコストがかかってしまう．しかし， :math:`\VStar` がスパースな行列であることに着目すると， :math:`\VStar` の逆行列は

.. math::
    {\VStar}^{-1} = \begin{bmatrix}
        {\VStar}^{-1}_{1} & 0 & 0 & 0 \\
        0 & {\VStar}^{-1}_{2} & 0 & 0 \\
        0 & 0 & {\VStar}^{-1}_{3} & 0 \\
        0 & 0 & 0 & {\VStar}^{-1}_{4} \\
    \end{bmatrix}
    :label: v-star-inv

となるため， :math:`\VStar_{i},i=1,\dots,m` のそれぞれについて逆行列を求めればよいことがわかる．結果として :math:`\VStar` の逆行列の計算量は視点数 :math:`m` に対して線型に増加することになり， :math:`\VStar` の逆行列を直接求めるのと比較すると計算量を一気に削減できる．

:math:`\Da` を求める際には， :math:`S = U^{*} - W{\VStar}^{-1}W^{\top}` の逆行列を :eq:`derivation-da` の両辺に左からかける必要がある．しかし，一般的にランドマーク数 :math:`n` よりもカメラの視点数 :math:`m` の方が圧倒的に小さい :math:`(m \ll n)` ため， :math:`S` のサイズは :math:`\VStar` と比べると圧倒的に小さい．したがって， :math:`S` の逆行列を求める処理は全体の計算量にはほとんど影響しない．

問題のサイズ(視点数や復元対象となるランドマークの数)が大きいときは， :eq:`lm-update` を直接解いて :math:`\DP` を得るよりも， :eq:`derivation-da` :eq:`derivation-db` :eq:`v-star-inv` によって :math:`\Da` と :math:`\Db` をそれぞれ計算し結合することで :math:`\DP` を得るほうが圧倒的に高速である．


改良
====

[#Agarwal_et_al_2010]_ は inexact Newton method とPCG(Preconditioned Conjugate Gradients)法を組み合わせることでより高速に更新量を求める手法を提案している．

SBAでは，誤差関数の更新則 :eq:`lm-update` を変形し， :eq:`derivation-da` :eq:`derivation-db` という2つの線型方程式を解く問題に落とし込んでいる．
このうち :eq:`derivation-db` は :math:`\VStar` のスパース性を利用して高速に解くことができたが， :eq:`derivation-da` は :math:`S` の逆行列を直接計算する必要があった．

SBAでは :eq:`derivation-da` と :eq:`derivation-db` を解くことで各iterationにおける"厳密な"更新量 :math:`\DP` を求めている．これに対して [#Agarwal_et_al_2010]_ は必ずしも :eq:`derivation-da` :eq:`derivation-db` の厳密な解を求める必要はなく，より高速な近似的計算によって厳密解を代替できることを主張している． すなわち，最終的な目的は誤差関数 :eq:`error-function` を十分小さくするような解を見つけることであり，もしそれが達成できるのであれば，必ずしも各ステップにおいて厳密な更新量を見つける必要はないのである．各iterationごとにより少ない計算量で近似的に更新量を求められれば，最適解に達するまでのステップ数が増えたとしても，全体の計算量を軽くすることができる可能性がある．

2通りのやり方がある．

    1. :eq:`lm-update` に直接PCG法を適用する
    2. :eq:`derivation-da` にPCG法を適用する



.. [#Lourakis_et_al_2015] Lourakis, Manolis IA, and Antonis A. Argyros. "SBA: A software package for generic sparse bundle adjustment." ACM Transactions on Mathematical Software (TOMS) 36.1 (2009): 2.
.. [#Gallego_et_al_2015] Gallego, Guillermo, and Anthony Yezzi. "A compact formula for the derivative of a 3-D rotation in exponential coordinates." Journal of Mathematical Imaging and Vision 51.3 (2015): 378-384.
.. [#Levenberg_1944] Levenberg, Kenneth. "A method for the solution of certain non-linear problems in least squares." Quarterly of applied mathematics 2.2 (1944): 164-168.
.. [#Coppersmith_et_al_1990] Coppersmith, Don, and Shmuel Winograd. "Matrix multiplication via arithmetic progressions." Journal of symbolic computation 9.3 (1990): 251-280.
.. [#Agarwal_et_al_2010] Agarwal, Sameer, et al. "Bundle adjustment in the large." European conference on computer vision. Springer, Berlin, Heidelberg, 2010.
