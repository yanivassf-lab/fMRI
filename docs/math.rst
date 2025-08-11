
=======================================
Functional PCA - Mathmatical Background
=======================================


Below is the step-by-step mathematical description of the method. The equations are presented at the level of a single voxel time series. (In the implementation, these calculations are applied in batches for efficiency.)

B-Spline Representation of the Signal
---------------------------------------
Let :math:`n` is number of the timepoints, and :math:`N` is the nuber of the voxels. Each voxel’s time series, denoted by

.. math::

   y(t) \in \mathbb{R}^{n},

is approximated using a spline basis with :math:`K` basis functions :math:`f_k(t)`:

.. math::

   y(t) \approx \sum_{k=1}^{K} c_k\, f_k(t),

or, in vector-matrix notation,

.. math::

   y \approx F\, c,

where:

- :math:`F \in \mathbb{R}^{n \times K}` is the basis matrix with entries
  :math:`F_{ik} = f_k(t_i)`,
- :math:`c \in \mathbb{R}^{K}` is the vector of coefficients.

Regularized Regression for Spline Coefficient Estimation
---------------------------------------------------------
To estimate the coefficient vector :math:`c`, the method minimizes the penalized least-squares criterion:

.. math::

   J(c) = \|y - F\, c\|^2 + \lambda\, c^\top P\, c,

where:

- :math:`\lambda` is the regularization parameter,
- :math:`P \in \mathbb{R}^{K \times K}` is the penalty matrix defined as

.. math::

   P_{kl} = \int_{T_{\min}}^{T_{\max}} f_k^{(d)}(t)\, f_l^{(d)}(t)\, dt,

with :math:`d` indicating the derivative order to be penalized (e.g., :math:`d=2` to penalize curvature).

The minimizer :math:`c` satisfies the penalized normal equations:

.. math::

   \left(F^\top F + \lambda P\right) c = F^\top y.

Selection of :math:`\lambda` for each voxel using Generalized Cross-Validation (GCV)
------------------------------------------------------------------------------------
For selection of :math:`\lambda`, the following is calculated separately for each voxel. The hat matrix :math:`H_\lambda \in \mathbb{R}^{n \times n}` is calculated

.. math::

   H_\lambda = F\, \left(F^\top F + \lambda P\right)^{-1} F^\top.


Then the generalized cross-validation (GCV) score for each candidate :math:`\lambda` is computed as:

.. math::

   \text{GCV}(\lambda) = \frac{n\, \| (I - H_\lambda) y \|^2}{\left[\text{trace}(I - H_\lambda)\right]^2},

where:

- :math:`I` is the :math:`n \times n` identity matrix,
- :math:`n` is the number of timepoints.

The optimal :math:`\lambda` is chosen as the value that minimizes the GCV score.

Functional Principal Component Analysis (fPCA)
-----------------------------------------------
After estimating the coefficient vector :math:`c` for each voxel, these coefficients are stored in a matrix

.. math::

   C \in \mathbb{R}^{N \times K},

where :math:`N` is the number of voxels.

The fPCA steps are as follows:

1. Centering the Coefficient Matrix:

   .. math::

      \tilde{C} = C - \mu,\quad \text{with}\quad \mu = \frac{1}{N} \sum_{i=1}^{N} C_{i,:}.

2. Compute the Penalized Covariance Matrix :math:`\Sigma \in \mathbb{R}^{K \times K}` in the Spline Basis:

   .. math::

      \Sigma = \frac{1}{N}\, \tilde{C}^\top\, \tilde{C}\, U,

where the Gram matrix :math:`U \in \mathbb{R}^{K \times K}` is defined as:

.. math::

   U_{kl} = \int_{T_{\min}}^{T_{\max}} f_k(t)\, f_l(t)\, dt,  k,l=1,2 ... K

3. Eigen-decomposition:

   Solve for the eigenvalues :math:`\gamma` and eigenvectors :math:`\phi`:

   .. math::

      \Sigma\, \phi = \gamma\, \phi.

   The eigenvalues are sorted in descending order.

4. Compute Voxel Scores for the Principal Components:

   For each voxel :math:`i`, the scores are computed as:

   .. math::

      \text{scores}_i = \tilde{c}_i\, U\, \phi.

   - The computed score indicates the contribution or importance of that voxel for the corresponding principal component.
   - Using the spatial information from the original brain mask, these voxel scores are then mapped back to a volume of the brain. This produces an importance map, saved as a file with the brain’s dimensions, where each voxel’s value indicates its contribution to the fPCA component.


5. Recover Eigenfunctions in the Time Domain:

   The eigenfunctions that describe temporal dynamics are given by:

   .. math::

      \psi = F\, \phi.

   The resulting function :math:`\psi(t)=F_{t,:}\,\cdot \, \phi` is plotted as a graph (intensity plot) that illustrates the time-course of the fPCA component across the entire brain.



