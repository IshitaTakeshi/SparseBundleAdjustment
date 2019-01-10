========================
LM法
========================


概要
----

LM法は，Gauss-Newton法と最急降下法の性質を組み合わせた非線形最小二乗法の一種である．

:math:`\mathbf{\beta}` をパラメータとするあるベクトル値関数 :math:`\mathbf{f}(\mathbf{\beta})` と目標値ベクトル :math:`\mathbf{y}` について，次で定義される誤差 :math:`d^{2}_{\Sigma}(\mathbf{y}, \mathbf{f}(\mathbf{\beta}))` を最小化するような :math:`\mathbf{\beta}` を見つける問題を考える．

.. math::
    d^{2}_{\Sigma}(\mathbf{y}, \mathbf{f}(\mathbf{\beta})) = (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))^{\top}\Sigma^{-1} (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))
    :label: error

LM法はGauss-Newton法と最急降下法を組み合わせた手法だと解釈することがすることができる．
:math:`\mathbf{J}` を関数 :math:`\mathbf{f}` のヤコビ行列 :math:`\frac{\partial \mathbf{f}}{\partial \beta}` ， :math:`\mathbf{\delta}` を :math:`\mathbf{\beta}` の更新量として，Gauss-Newton法，最急降下法，LM法それぞれによる :math:`\mathbf{\delta}` の方法を示す．

.. math::
    \begin{align}
    \mathbf{\delta}_{GN}
    &= (\mathbf{J}^{\top} \Sigma^{-1} \mathbf{J})^{-1}
       \mathbf{J}^{\top} \Sigma^{-1} [\mathbf{y} - \mathbf{f}(\mathbf{\beta})] \\
    \mathbf{\delta}_{GD}
    &= \mathbf{J}^{\top} \Sigma^{-1} [\mathbf{y} - \mathbf{f}(\mathbf{\beta})] \\
    \mathbf{\delta}_{LM}
    &= (\mathbf{J}^{\top} \Sigma^{-1} \mathbf{J} + \lambda \mathbf{I})^{-1}
       \mathbf{J}^{\top} \Sigma^{-1} [\mathbf{y} - \mathbf{f}(\mathbf{\beta})]
    \end{align}

:math:`\mathbf{I}` は単位行列であり， :math:`\lambda \in \mathbb{R}, \lambda > 0` は damping parameter と呼ばれる値である．

それぞれの式を見比べると，

- LM法による更新量の計算方法はGauss-Newton法と最急降下法を組み合わせたものである
- Gauss-Newton法と最急降下法のどちらの性質を強くするかを damping parameter がコントロールしている

ということがわかる．

導出
----

:math:`\Sigma` を分散共分散行列とし，誤差をmahalanobis距離によって次のように定義する．

.. math::
    d^{2}_{\Sigma}(\mathbf{y}, \mathbf{f}(\mathbf{\beta} + \mathbf{\delta})) = (\mathbf{y} - \mathbf{f}(\mathbf{\beta} + \mathbf{\delta}))^{\top}\Sigma^{-1} (\mathbf{y} - \mathbf{f}(\mathbf{\beta} + \mathbf{\delta}))
    :label: updated_error


関数 :math:`\mathbf{f}` を :math:`\mathbf{f}(\mathbf{\beta} + \mathbf{\delta}) \approx \mathbf{f}(\mathbf{\beta}) + \mathbf{J} \mathbf{\delta}` と近似すると， :eq:`updated_error` は

.. math::
    \begin{align}
    d^{2}_{\Sigma}(\mathbf{y}, \mathbf{f}(\mathbf{\beta} + \mathbf{\delta}))
    &\approx (\mathbf{y} - \mathbf{f}(\mathbf{\beta}) - \mathbf{J}\mathbf{\delta})^{\top} \Sigma^{-1} (\mathbf{y} - \mathbf{f}(\mathbf{\beta}) - \mathbf{J}\mathbf{\delta}) \\
    &= (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))^{\top} \Sigma^{-1}  (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))
    - 2 (\mathbf{y} - \mathbf{f}(\mathbf{\beta}))^{\top} \Sigma^{-1} \mathbf{J} \mathbf{\delta}
    + \mathbf{\delta}^{\top} \mathbf{J}^{\top} \Sigma^{-1} \mathbf{J} \mathbf{\delta}
    \end{align}


となる．これを :math:`\mathbf{\delta}` で微分して :math:`\mathbf{0}` とおくと，

.. math::
    \mathbf{J}^{\top} \Sigma^{-1} \mathbf{J} \mathbf{\delta}
    = \mathbf{J}^{\top} \Sigma^{-1} [\mathbf{y} - \mathbf{f}(\mathbf{\beta})]

が得られる．左辺に :math:`\lambda \mathbf{I}` という項を組み込んでしまえば，即座にLM法が得られる．

.. math::
    (\mathbf{J}^{\top} \Sigma^{-1} \mathbf{J} + \lambda \mathbf{I}) \mathbf{\delta}
    = \mathbf{J}^{\top} \Sigma^{-1} [\mathbf{y} - \mathbf{f}(\mathbf{\beta})]
