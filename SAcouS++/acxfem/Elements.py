from functools import cached_property
import numpy as np

from .Basis import Lobbato1DElement, Lagrange2DTriElement, Lagrange2DQuad
from .PrecomputeMatrices import Ke1D, Me1D, Ce1D


class Helmholtz1DElement(Lobbato1DElement):

  def __init__(self, label, order, nodes, mat_coeffs=[]):
    super().__init__(label, order, nodes)
    self.mat_coeffs = mat_coeffs

  @cached_property
  def ke(self):
    """compute the elementary stiffness matrix
    returns:
    K: ndarray
        elementary stiffness matrix
    """
    Ke = 0
    if self.order == 1:
      Ke = self.inverse_Jacobian * self.mat_coeffs[0] * Ke1D[0]
    elif self.order == 2:
      Ke = self.inverse_Jacobian * self.mat_coeffs[0] * Ke1D[1]
    elif self.order == 3:
      Ke = self.inverse_Jacobian * self.mat_coeffs[0] * Ke1D[2]
    elif self.order == 4:
      Ke = self.inverse_Jacobian * self.mat_coeffs[0] * Ke1D[3]
    else:
      print("quadrtic lobatto not supported yet")
    return Ke

  @cached_property
  def me(self):
    """compute the elementary stiffness matrix
    returns:
    m: ndarray
        elementary stiffness matrix
    """
    if self.order == 1:
      Me = self.Jacobian * self.mat_coeffs[1] * Me1D[0]
    elif self.order == 2:
      Me = self.Jacobian * self.mat_coeffs[1] * Me1D[1]
    elif self.order == 3:
      Me = self.Jacobian * self.mat_coeffs[1] * Me1D[2]
    elif self.order == 4:
      Me = self.Jacobian * self.mat_coeffs[1] * Me1D[3]
    else:
      print("quadrtic lobatto not supported yet")
    return Me

  @cached_property
  def ce(self):
    """compute the elementary coupling matrix, N(x)B(x)
    returns:
    c: ndarray
        elementary coupling matrix
    """
    if self.order == 1:
      Ce = self.mat_coeffs[2] * Ce1D[0]
    elif self.order == 2:
      Ce = self.mat_coeffs[2] * Ce1D[1]
    elif self.order == 3:
      Ce = self.mat_coeffs[2] * Ce1D[2]
    elif self.order == 4:
      Ce = self.mat_coeffs[2] * Ce1D[3]
    else:
      print("quadrtic lobatto not supported yet")
    return Ce


# consider material properties: ease the matrix assembly
class Helmholtz2DElement(Lagrange2DTriElement):

  def __init__(self, label, order, vertices, mat_coeffs=[]):
    super().__init__(label, order, vertices)
    self.mat_coeffs = mat_coeffs

  @cached_property
  def ke(self):
    """compute the elementary stiffness matrix
    returns:
    K: ndarray
        elementary stiffness matrix
    """
    if self.order == 1:
      Ke = self.mat_coeffs[0] * sum(
          self.B[i, :, :] @ self.inv_J_product @ self.B[i, :, :].T * weight
          for i, weight in enumerate(self.weights)) * self.det_J

    else:
      print("quadrtic lagrange not implemented yet")

    return Ke

  @cached_property
  def me(self):
    """compute the elementary stiffness matrix
    returns:
    m: ndarray
        elementary stiffness matrix
    """
    if self.order == 1:
      weight = np.diag(
          np.array([self.weights[0], self.weights[1], self.weights[2]]))
      Me = self.mat_coeffs[
          1] * self.N[:, :].T @ weight @ self.N[:, :] * self.det_J
    else:
      print("quadrtic lagrange not implemented yet")
    return Me


from .Quadratures import GaussLegendre2DQuad


class GeneralShellElement:

  def __init__(self, label, order, vertices):
    """
    MITC4 shell element: Mixed Interpolation of Tensorial Components
    parameters:
    label: str
        label of the element
    order: int
        order of the element
    vertices: ndarray
        vertices coordinates
    mat_coeffs: list
    """
    super().__init__(label, order, vertices)
    parent_element = Lagrange2DQuad(1, 1)
    gl_points, gm_weights = GaussLegendre2DQuad(
        1).points(), GaussLegendre2DQuad(1).weights()
    self.dN_dxi_eta = self.parent_element.get_der_shape_functions(xi, eta)

  def Jacobian(self, xi, eta):
    J = dN_dxi_eta @ node_coords
    return J, np.linalg.det(J), np.linalg.inv(J)

  def shape_function(self, xi, eta):
    """compute the shape functions at the given xi, eta
    parameters:
    xi: float
        xi coordinate
    eta: float
        eta coordinate
    returns:
    N: ndarray
        shape functions
    """
    N1 = 0.25 * (1 - xi) * (1 - eta)
    N2 = 0.25 * (1 + xi) * (1 - eta)
    N3 = 0.25 * (1 + xi) * (1 + eta)
    N4 = 0.25 * (1 - xi) * (1 + eta)
    return np.array([N1, N2, N3, N4])

  def jacobian(self, xi, eta, node_coords):
    J = dN_dxi_eta @ node_coords
    return J, np.linalg.det(J), np.linalg.inv(J)
