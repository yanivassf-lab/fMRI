Functional PCA: mathematical background
================================================================================

This page explains the mathematical method. The fPCA method follows the work of Roberto Viviani, Georg Grön and Manfred Spitzer, `Functional Principal Component Analysis of fMRI Data, Human Brain Mapping 24, 109–129 (2005) <https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.20074>`_.
The calculation used by the program is described separately in :doc:`implementation`.

There are :math:`n` time points and :math:`N` voxels. For one voxel,
:math:`y(t)` is a signal as a function of time. Its recorded values form the
column vector :math:`y=(y(t_1),\ldots,y(t_n))^\top`.

Represent the signal with smooth curves
--------------------------------------------------------------------------------

We use :math:`K` B-spline basis functions. These are simple curves that can
be added together to describe a smooth signal:

.. math::

   y(t) \approx \sum_{k=1}^{K} c_k f_k(t).

The coefficients :math:`c_k` control the contribution of each basis function.
At the recorded time points, the same expression is

.. math::

   y \approx Fc, \qquad F_{jk}=f_k(t_j),

where :math:`F` has size :math:`n\times K` and :math:`c` has size
:math:`K\times 1`.

Fit the curve while limiting roughness
--------------------------------------------------------------------------------

A curve should follow the data without following every small fluctuation.
We choose its coefficients by minimizing

.. math::

   J(c)=\lVert y-Fc\rVert^2+\lambda c^\top P c,

with

.. math::

   P_{kl}=\int_{T_{\min}}^{T_{\max}}
   f_k^{(d)}(t)f_l^{(d)}(t)\,dt.

The first term measures the difference from the recorded signal. The second
term penalizes roughness. With :math:`d=2`, it penalizes curvature, which
measures how quickly the slope changes.

The parameter :math:`\lambda` controls the balance between the two terms.
A larger value gives more weight to smoothness. It does not force the
signal to be zero: constant and linear functions have zero curvature.

The fitted coefficients satisfy

.. math::

   (F^\top F+\lambda P)c=F^\top y.

When the matrix on the left is invertible, this equation has a unique solution.

Choose the smoothing level for each voxel
--------------------------------------------------------------------------------

Generalized cross-validation (GCV) balances the error of the fit against
the flexibility of the fitted curve. Define

.. math::

   H_\lambda=F(F^\top F+\lambda P)^{-1}F^\top.

Then :math:`H_\lambda y` is the fitted signal, and
:math:`\operatorname{tr}(H_\lambda)` measures its effective flexibility.
For each candidate smoothing level, calculate

.. math::

   \operatorname{GCV}(\lambda)=
   \frac{n\lVert(I-H_\lambda)y\rVert^2}
        {[n-\operatorname{tr}(H_\lambda)]^2}.

Choose the candidate with the smallest score, separately for each voxel.
The denominator must be nonzero.

This step chooses the smoothness of a time series. It is different from
cross-validation of a classifier across participants. In particular, it
does not test prediction on new participants.

Center the fitted curves
--------------------------------------------------------------------------------

Collect the fitted coefficients in a matrix :math:`C` of size
:math:`N\times K`, with one row per voxel. Let :math:`c_i` be the coefficient
column vector for voxel :math:`i`.

.. math::

   \mu_c=\frac{1}{N}\sum_{i=1}^{N}c_i,
   \qquad \widetilde c_i=c_i-\mu_c,
   \qquad S=\frac{1}{N}\widetilde C^\top\widetilde C.

The mean curve is

.. math::

   \mu(t)=\sum_{k=1}^{K}(\mu_c)_k f_k(t).

Centering removes this mean curve across voxels. It does not mean subtracting
each voxel's own average over time.

Find the main temporal patterns
--------------------------------------------------------------------------------

B-spline basis functions are not generally orthogonal. We therefore need
their Gram matrix, which records their overlap:

.. math::

   U_{kl}=\int_{T_{\min}}^{T_{\max}}f_k(t)f_l(t)\,dt.

The fPCA coefficient vectors solve

.. math::

   SU\phi_j=\gamma_j\phi_j,
   \qquad \phi_j^\top U\phi_k=\delta_{jk}.

Here, :math:`\delta_{jk}` is 1 when :math:`j=k` and 0 otherwise. This
condition makes the temporal patterns have unit length and be orthogonal
under integration over time.

The eigenvalues :math:`\gamma_j` measure the variance explained by the
patterns. We order them from largest to smallest. The temporal pattern
for component :math:`j` is

.. math::

   \psi_j(t)=\sum_{k=1}^{K}(\phi_j)_k f_k(t).

At the recorded time points, its values are :math:`F\phi_j`.

The matrices :math:`S` and :math:`U` are symmetric, but their product
:math:`SU` need not be symmetric. The Gram matrix accounts for the basis
overlap; it is not an additional smoothing penalty.

Find voxel scores and reconstruct a signal
--------------------------------------------------------------------------------

The score of voxel :math:`i` on component :math:`j` is

.. math::

   a_{ij}=\widetilde c_i^\top U\phi_j.

A score may be positive or negative. Mapping it back to the voxel's
location gives the spatial score map for that component.

With :math:`r` retained components, the fitted signal is approximated by

.. math::

   \widehat y_i(t)\approx\mu(t)+\sum_{j=1}^{r}a_{ij}\psi_j(t).

The sum without :math:`\mu(t)` describes the deviation from the spatial
mean curve. The full fitted curve and this centered representation serve
different purposes.

Changing the signs of both :math:`\phi_j` and its scores leaves the
reconstruction unchanged. Component signs alone therefore do not establish
the direction of a biological effect.

Reference
--------------------------------------------------------------------------------

Roberto Viviani, Georg Grön, and Manfred Spitzer. *Functional Principal
Component Analysis of fMRI Data*. Human Brain Mapping 24, 109–129 (2005).

