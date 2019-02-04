====================
投影モデルとその微分
====================


3次元点 :math:`\mathbf{p} = \begin{bmatrix} X & Y & Z \end{bmatrix}^{\top}` を2次元に投影するモデルは :eq:`projection_model` で表される．

.. math::
    \hat{\mathbf{x}} = \mathbf{\pi}(\mathrm{R}(\mathbf{\omega})\mathbf{b} + \mathbf{t})
    :label: projection_model


ここで， :math:`\mathrm{K}` はカメラの内部行列， :math:`\mathrm{\mathrm{R}(\mathbf{\omega})}` はリー代数の元 :math:`\mathbf{\omega}` に対応する回転行列であり， :math:`\mathbf{\pi}` はカメラ座標系に変換された3次元点を画像平面上に投影する関数である．


.. math::
    \mathbf{\pi}(\begin{bmatrix} X \\ Y \\ Z \end{bmatrix})
    = \frac{1}{Z}
    \begin{bmatrix}
        X \\ Y
    \end{bmatrix}


投影モデルのヤコビ行列は次のようになる．


.. math::
    \frac{\partial \mathbf{\pi}(\mathbf{p})}{\partial \mathbf{p}}
    = \begin{bmatrix}
        \frac{1}{Z} &           0 & -\frac{X}{Z^{2}} \\
                  0 & \frac{1}{Z} & -\frac{Y}{Z^{2}}
    \end{bmatrix}
    :label: projection_jacobian
