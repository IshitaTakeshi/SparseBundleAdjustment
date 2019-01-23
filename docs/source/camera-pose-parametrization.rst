================
カメラの回転表現
================

.. math::
    \def\mbf#1{{\mathbf #1}}
    \def\SO(#1){{\mathrm{SO}(#1)}}
    \def\so(#1){{\mathfrak{so}(#1)}}
    \def\SE(#1){{\mathrm{SE}(#1)}}
    \def\skew#1{{\left[ #1 \right]_{\times}}}


SfMは，カメラから得た視覚情報を用いて，各視点におけるカメラの姿勢とランドマークの3次元座標を求める問題である．
SLAMはカメラから逐次的に得られる視覚情報を用いて環境地図とその中の相対的なカメラ姿勢を求める問題である．

| これらの問題では，カメラ姿勢とランドマークの3次元座標をパラメータとして最適化問題を解くアプローチがしばしば用いられる．このうちカメラ姿勢は原点座標からの並進と基準姿勢からの回転の組み合わせで表現することができる．このときカメラの回転を :math:`3 \times 3` の行列で表現して最適化問題を解くと，たいていの場合最適化の結果として回転行列でない行列 :math:`A \in \mathbb{R}^{3 \times 3} \setminus \SO(3)` が得られてしまう．
| ここで :math:`\SO(3)` は回転行列の集合である．

.. math::
    \SO(3) = \left\{
        R \in \mathbb{R}^{3 \times 3} \mid R^{\top}R = I, \det(R) = 1
    \right\}
    :label: SO3-group-definition

したがって，SfMやSLAMでは，回転行列の代わりにできる限り簡潔なパラメータ :math:`\mbf{\omega} \in \mathbb{S}` でカメラの回転を表現し，最適化の結果として得られた :math:`\hat{\mbf{\omega}}` をある関数 :math:`\phi: \mathbb{S} \to \SO(3)` で射影することで回転行列 :math:`\hat{R} \in \SO(3)` を得る．

回転を表現するためのパラメータ集合 :math:`\mathbb{S}` としては :math:`\SO(3)` のリー代数 :math:`\so(3)` や単位四元数の集合 :math:`\left\{\mbf{q} \in \mathbb{H} \mid ||q|| = 1 \right\}` が用いられる．本章では前者について解説する．


:math:`\so(3)` による回転表現
=============================

ここでは回転行列 :math:`R \in \SO(3)` が3次元ベクトル :math:`\mbf{\omega} \in \mathbb{R}^{3}` から生成できることを示す．

:math:`\so(3)` は以下で定義される集合である．

.. math::
    \so(3) = \left\{
        \skew{\mbf{\omega}} \mid \mbf{\omega} \in \mathbb{R}^{3}
    \right\}

ここで :math:`\skew{\cdot}` は3次元ベクトルに対応する歪対称行列を表現する演算子である．

.. math::
    \skew{\mbf{\omega}} = \begin{bmatrix}
        0 & -\omega_{3} & \omega_{2} \\
        \omega_{1} & 0 & -\omega_{1} \\
        -\omega_{2} & \omega_{3} & 0
    \end{bmatrix}

:math:`\so(3)` の元を指数写像で射影すると回転行列が得られる．

.. math::
    \exp : \so(3) \to \SO(3)


導出
~~~~

ここでは回転群の性質から出発して :math:`\so(3)` を導出し，さらに :math:`\so(3)` を指数写像で射影すると :math:`\SO(3)` が得られることを見る．

実数から回転行列への写像 :math:`R(t)` を考える．ただし :math:`t=0` において :math:`R(t) = I` を通るものとする．


.. math::
    R(t) : \mathbb{R} \to \SO(3), \; R(0) = I


回転行列の定義から， :math:`R(t)` は次を満たす．


.. math::
    R(t) R(t)^{\top} = I
    :label: orthogonality


微分すると


.. math::
    \frac{d}{dt} (R(t)R(t)^{\top})
    = \frac{d R(t)}{dt} R(t)^{\top} + R(t)(\frac{d R(t)}{dt})^{\top}
    = 0

.. math::
    \begin{align}
        \frac{d R(t)}{dt} R(t)^{\top}
        &= -R(t)(\frac{d R(t)}{dt})^{\top} \\
        &= -(\frac{d R(t)}{dt} R(t)^{\top})^{\top}
    \end{align}


したがって :math:`\frac{d R(t)}{dt} R(t)^{\top}` は歪対称行列であり， :math:`\omega \in \mathbb{R}^{3}` を用いて


.. math::
    \frac{d R(t)}{dt} R(t)^{\top} = \skew{\mbf{\omega}}
    :label: differential-equation-so3

| と表すことができる．
| :math:`R` を3つの直交基底 :math:`\begin{bmatrix} \mbf{e}_{1}(t) & \mbf{e}_{2}(t) & \mbf{e}_{3}(t) \end{bmatrix}` で表現すると， :eq:`differential-equation-so3` は


.. math::
    \frac{d \mbf{e}_{i}(t)}{dt} = \skew{\mbf{\omega}(t)} \, \mbf{e}_{i}(t),\; i = 1,\dots,3


という微分方程式であることがわかる．この方程式の解は


.. math::
    \mbf{e}_{i}(t) = \exp(\skew{\mbf{\omega}(t)} t) \, \mbf{e}_{i}(0),\; i = 1,\dots,3


であることから， :math:`R(t)` は :math:`\mbf{\omega}` を用いて


.. math::
    \begin{align}
        R(t) &= \exp(\skew{\mbf{\omega}(t)} t) \, R(0)  \\
             &= \exp(\skew{\mbf{\omega}(t)} t)
    \end{align}
    :label: exponential-map


と表現することができる．
さて，:math:`R(t)` の指数写像による生成方法 :eq:`exponential-map` は行列の直交性 :eq:`orthogonality` のみから導かれたため， :math:`\det(R(t)) = 1` を示さなければ :math:`R(t)` が真に :math:`\SO(3)` の元であるということは言えない．
しかし，指数写像 :eq:`exponential-map` によって得られた :math:`R(t)` が :math:`\det(R(t)) = 1` を充足することは簡単に示すことができる．

正方行列 :math:`A` について :math:`\det(\exp(A))=\exp({\operatorname{tr} (A)})` が成り立つことから，

.. math::
    \begin{align}
        \det(R(t))
        &= \det(\exp(\skew{\mbf{\omega}} t)) \\
        &= \exp(\operatorname{tr}(\skew{\mbf{\omega}} t)) \\
        &= \exp(0) \\
        &= 1
    \end{align}

となり， :math:`R(t)` はやはり :math:`\SO(3)` の元であることがわかる．


指数写像
~~~~~~~~


Rodriguesの回転公式
~~~~~~~~~~~~~~~~~~~

指数写像を実装するわけにはいかないため，代わりにRodriguesの回転公式を用いる．
