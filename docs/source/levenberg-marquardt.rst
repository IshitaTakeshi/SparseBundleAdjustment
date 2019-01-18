========================
LM法
========================


概要
----

ニュートン法は収束性が保証されておらず，問題によっては解を見つけられないことがある．LM法はニュートン法と最急降下法を組み合わせることで収束性を保証したアルゴリズムである [#Wright_et_al_1999]_ ．

:math:`\mathbf{\beta}` をパラメータとするあるベクトル値関数 :math:`\mathbf{f}(\mathbf{\beta})` と目標値ベクトル :math:`\mathbf{y}` について，次で定義される誤差 :math:`d^{2}_{\mathrm{\Sigma}}(\mathbf{y}, \mathbf{f}(\mathbf{\beta}))` を最小化するような :math:`\mathbf{\beta}` を見つける問題を考える．

.. math::
    d^{2}_{\mathrm{\Sigma}}(\mathbf{y}, \mathbf{f}(\mathbf{\beta})) = (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))^{\top}\mathrm{\Sigma}^{-1} (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))
    :label: error

LM法はGauss-Newton法と最急降下法を組み合わせた手法だと解釈することがすることができる．
:math:`\mathrm{J}` を関数 :math:`\mathbf{f}` のヤコビ行列 :math:`\frac{\partial \mathbf{f}}{\partial \beta}` ， :math:`\mathbf{\delta}` を :math:`\mathbf{\beta}` の更新量として，Gauss-Newton法，最急降下法，LM法それぞれによる :math:`\mathbf{\delta}` の方法を示す．

.. math::
    \begin{align}
    \mathbf{\delta}_{GN}
    &= (\mathrm{J}^{\top} \mathrm{\Sigma}^{-1} \mathrm{J})^{-1}
       \mathrm{J}^{\top} \mathrm{\Sigma}^{-1} [\mathbf{y} - \mathbf{f}(\mathbf{\beta})] \\
    \mathbf{\delta}_{GD}
    &= \mathrm{J}^{\top} \mathrm{\Sigma}^{-1} [\mathbf{y} - \mathbf{f}(\mathbf{\beta})] \\
    \mathbf{\delta}_{LM}
    &= (\mathrm{J}^{\top} \mathrm{\Sigma}^{-1} \mathrm{J} + \lambda \mathrm{I})^{-1}
       \mathrm{J}^{\top} \mathrm{\Sigma}^{-1} [\mathbf{y} - \mathbf{f}(\mathbf{\beta})]
    \end{align}

:math:`\mathrm{I}` は単位行列であり， :math:`\lambda \in \mathbb{R}, \lambda > 0` は damping parameter と呼ばれる値である．

それぞれの式を見比べると，

- LM法による更新量の計算方法はGauss-Newton法と最急降下法を組み合わせたものである
- Gauss-Newton法と最急降下法のどちらの性質を強くするかを damping parameter がコントロールしている

ということがわかる．

導出
----

:math:`\mathrm{\Sigma}` を分散共分散行列とし，誤差をmahalanobis距離によって次のように定義する．

.. math::
    d^{2}_{\mathrm{\Sigma}}(\mathbf{y}, \mathbf{f}(\mathbf{\beta} + \mathbf{\delta})) = (\mathbf{y} - \mathbf{f}(\mathbf{\beta} + \mathbf{\delta}))^{\top}\mathrm{\Sigma}^{-1} (\mathbf{y} - \mathbf{f}(\mathbf{\beta} + \mathbf{\delta}))
    :label: updated-error


関数 :math:`\mathbf{f}` を :math:`\mathbf{f}(\mathbf{\beta} + \mathbf{\delta}) \approx \mathbf{f}(\mathbf{\beta}) + \mathrm{J} \mathbf{\delta}` と近似すると， :eq:`updated-error` は

.. math::
    \begin{align}
    d^{2}_{\mathrm{\Sigma}}(\mathbf{y}, \mathbf{f}(\mathbf{\beta} + \mathbf{\delta}))
    &\approx (\mathbf{y} - \mathbf{f}(\mathbf{\beta}) - \mathrm{J}\mathbf{\delta})^{\top} \mathrm{\Sigma}^{-1} (\mathbf{y} - \mathbf{f}(\mathbf{\beta}) - \mathrm{J}\mathbf{\delta}) \\
    &= (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))^{\top} \mathrm{\Sigma}^{-1}  (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))
    - 2 (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))^{\top} \mathrm{\Sigma}^{-1} \mathrm{J} \mathbf{\delta}
    + \mathbf{\delta}^{\top} \mathrm{J}^{\top} \mathrm{\Sigma}^{-1} \mathrm{J} \mathbf{\delta}
    \end{align}


となる．これを :math:`\mathbf{\delta}` で微分して :math:`\mathbf{0}` とおくと，

.. math::
    \mathrm{J}^{\top} \mathrm{\Sigma}^{-1} \mathrm{J} \mathbf{\delta}
    = \mathrm{J}^{\top} \mathrm{\Sigma}^{-1} [\mathbf{y} - \mathbf{f}(\mathbf{\beta})]

が得られる．左辺に :math:`\lambda \mathrm{I}` という項を組み込んでしまえば，即座にLM法が得られる．

.. math::
    (\mathrm{J}^{\top} \mathrm{\Sigma}^{-1} \mathrm{J} + \lambda \mathrm{I}) \mathbf{\delta}
    = \mathrm{J}^{\top} \mathrm{\Sigma}^{-1} [\mathbf{y} - \mathbf{f}(\mathbf{\beta})]


反復アルゴリズム
----------------

LM法
~~~~

.. [#Wright_et_al_1999] Wright, Stephen, and Jorge Nocedal. "Numerical optimization." Springer Science 35.67-68 (1999): 7.
