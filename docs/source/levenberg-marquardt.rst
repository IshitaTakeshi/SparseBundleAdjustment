========================
LM法
========================

.. math::
    \def\B{{\mathbf{\beta}}}
    \def\D{{\mathbf{\delta}}}

概要
----

勾配の2次微分の情報を利用する最適化手法の一種Gauss-Newton法は収束性が保証されていない．LM法はGauss-Newton法と最急降下法を組み合わせることで収束性を保証したアルゴリズムである [#Wright_et_al_1999]_ ．

:math:`\B` をパラメータとするあるベクトル値関数 :math:`\mathbf{f}(\B)` と目標値ベクトル :math:`\mathbf{y}` について，次で定義される誤差 :math:`d^{2}_{\Sigma}(\mathbf{y}, \mathbf{f}(\B))` を最小化するような :math:`\B` を見つける問題を考える．

.. math::
    d^{2}_{\Sigma}(\mathbf{y}, \mathbf{f}(\B)) = (\mathbf{y} - \mathbf{f}(\B))^{\top}\Sigma^{-1} (\mathbf{y} - \mathbf{f}(\B))
    :label: error

LM法はGauss-Newton法と最急降下法を組み合わせた手法だと解釈することがすることができる．
:math:`J` を関数 :math:`\mathbf{f}` のヤコビ行列 :math:`\frac{\partial \mathbf{f}}{\partial \beta}` ， :math:`\D` を :math:`\B` の更新量として，Gauss-Newton法，最急降下法，LM法それぞれによる :math:`\D` の方法を示す．

.. math::
    \begin{align}
    \D_{GN}
    &= (J^{\top} \Sigma^{-1} J)^{-1}
       J^{\top} \Sigma^{-1} [\mathbf{y} - \mathbf{f}(\B)] \\
    \D_{GD}
    &= J^{\top} \Sigma^{-1} [\mathbf{y} - \mathbf{f}(\B)] \\
    \D_{LM}
    &= (J^{\top} \Sigma^{-1} J + \lambda I)^{-1}
       J^{\top} \Sigma^{-1} [\mathbf{y} - \mathbf{f}(\B)]
    \end{align}

:math:`I` は単位行列であり， :math:`\lambda \in \mathbb{R}, \lambda > 0` は damping parameter と呼ばれる値である．

それぞれの式を見比べると，

- LM法による更新量の計算方法はGauss-Newton法と最急降下法を組み合わせたものである
- Gauss-Newton法と最急降下法のどちらの性質を強くするかを damping parameter がコントロールしている

ということがわかる．Damping parameter を大きくすると最急降下法の性質が強くなり，小さくするとGauss-Newton法の性質が強くなる(誤差が発散する可能性が高くなる)．

時刻 :math:`t` におけるパラメータ :math:`\B` の値を :math:`\B^{(t)}` とする．このとき，LM法は次に示す規則にしたがってパラメータ :math:`\B` を更新する．

- 誤差が減少する :math:`\left( f(\B^{(t)} + \D) < f(\B^{(t)}) \right)` ならばパラメータを :math:`\B^{(t+1)} \leftarrow \B^{(t)} + \D` と更新する．
- 誤差が減少しない :math:`\left( f(\B^{(t)} + \D) \geq f(\B^{(t)}) \right)` ならば :math:`\lambda` の値を大きくし，再度更新量 :math:`\D` を計算し直す．誤差が減少するような :math:`\D` が見つかるまでこれを繰り返す．

LM法は，damping parameter を変化させながら誤差が必ず減少するような更新量 :math:`\D` を探し出すことで，誤差の収束を保証している．


導出
----

:math:`\Sigma` を分散共分散行列とし，誤差をmahalanobis距離によって次のように定義する．

.. math::
    d^{2}_{\Sigma}(\mathbf{y}, \mathbf{f}(\B + \D)) = (\mathbf{y} - \mathbf{f}(\B + \D))^{\top}\Sigma^{-1} (\mathbf{y} - \mathbf{f}(\B + \D))
    :label: updated-error


関数 :math:`\mathbf{f}` を :math:`\mathbf{f}(\B + \D) \approx \mathbf{f}(\B) + J \D` と近似すると， :eq:`updated-error` は

.. math::
    \begin{align}
    d^{2}_{\Sigma}(\mathbf{y}, \mathbf{f}(\B + \D))
    &\approx (\mathbf{y} - \mathbf{f}(\B) - J\D)^{\top} \Sigma^{-1} (\mathbf{y} - \mathbf{f}(\B) - J\D) \\
    &= (\mathbf{y} - \mathbf{f}(\B))^{\top} \Sigma^{-1}  (\mathbf{y} - \mathbf{f}(\B))
    - 2 (\mathbf{y} - \mathbf{f}(\B))^{\top} \Sigma^{-1} J \D
    + \D^{\top} J^{\top} \Sigma^{-1} J \D
    \end{align}


となる．これを :math:`\D` で微分して :math:`\mathbf{0}` とおくと，

.. math::
    J^{\top} \Sigma^{-1} J \D
    = J^{\top} \Sigma^{-1} [\mathbf{y} - \mathbf{f}(\B)]

が得られる．左辺に :math:`\lambda I` という項を組み込んでしまえば，即座にLM法が得られる．

.. math::
    (J^{\top} \Sigma^{-1} J + \lambda I) \D
    = J^{\top} \Sigma^{-1} [\mathbf{y} - \mathbf{f}(\B)]



.. [#Wright_et_al_1999] Wright, Stephen, and Jorge Nocedal. "Numerical optimization." Springer Science 35.67-68 (1999): 7.
