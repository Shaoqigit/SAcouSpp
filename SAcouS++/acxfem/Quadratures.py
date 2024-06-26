# This file is part of PyXfem, a software distributed under the MIT license.
# For any question, please contact the authors cited below.
#
# Copyright (c) 2023
# 	Shaoqi WU <shaoqiwu@outlook.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# store legendre gauss quadrature weights and points

from abc import abstractmethod, ABCMeta
import numpy as np


class NumericalQuadrature(metaclass=ABCMeta):
  """base abstract numerical quadrature class
    """

  def __init__(self, n):
    """constructor

        Args:
            n (int): number of quadrature points
        """
    self.n = n

  @abstractmethod
  def points(self):
    pass

  @abstractmethod
  def weights(self):
    pass


class GaussLegendreQuadrature(NumericalQuadrature):
  """Gauss-Legendre quadrature
    """

  def __init__(self, n):
    """constructor

        Args:
            n (int): number of quadrature points
        """
    self.n = n

  def get_pts_wts(self):
    """compute quadrature points and weights
        """
    return self.points(self.n), self.weights(self.n)

  def points(self):
    """get quadrature points

        Returns:
            numpy.ndarray: quadrature points
        """
    if self.n == 1:
      return np.array([0.0])
    elif self.n == 2:
      return np.array([5.77350269189625731e-01, -5.77350269189625731e-01])
    elif self.n == 3:
      return np.array([
          0.00000000000000000e+00, 7.74596669241483404e-01,
          -7.74596669241483404e-01
      ])
    elif self.n == 4:
      return np.array([
          3.39981043584856257e-01, -3.39981043584856257e-01,
          8.61136311594052573e-01, -8.61136311594052573e-01
      ])
    elif self.n == 5:
      return np.array([
          0.00000000000000000e+00, 5.38469310105683108e-01,
          -5.38469310105683108e-01, 9.06179845938663964e-01,
          -9.06179845938663964e-01
      ])
    elif self.n == 6:
      return np.array([
          2.38619186083196905e-01, -2.38619186083196905e-01,
          6.61209386466264482e-01, -6.61209386466264482e-01,
          9.32469514203152050e-01, -9.32469514203152050e-01
      ])
    elif self.n == 7:
      return np.array([
          0.00000000000000000e+00, 4.05845151377397129e-01,
          -4.05845151377397129e-01, 7.41531185599394460e-01,
          -7.41531185599394460e-01, 9.49107912342758486e-01,
          -9.49107912342758486e-01
      ])
    elif self.n == 8:
      return np.array([
          1.83434642495649808e-01, -1.83434642495649808e-01,
          5.25532409916328991e-01, -5.25532409916328991e-01,
          7.96666477413626728e-01, -7.96666477413626728e-01,
          9.60289856497536287e-01, -9.60289856497536287e-01
      ])
    elif self.n == 9:
      return np.array([
          0.00000000000000000e+00, 3.24253423403808916e-01,
          -3.24253423403808916e-01, 6.13371432700590358e-01,
          -6.13371432700590358e-01, 8.36031107326635770e-01,
          -8.36031107326635770e-01, 9.68160239507626086e-01,
          -9.68160239507626086e-01
      ])
    elif self.n == 10:
      return np.array([
          1.48874338981631216e-01, -1.48874338981631216e-01,
          4.33395394129247213e-01, -4.33395394129247213e-01,
          6.79409568299024436e-01, -6.79409568299024436e-01,
          8.65063366688984536e-01, -8.65063366688984536e-01,
          9.73906528517171743e-01, -9.73906528517171743e-01
      ])

  def weights(self):
    """get quadrature weights
        Returns:"""
    if self.n == 1:
      return np.array([2.000000000000000e+00])
    elif self.n == 2:
      return np.array([1.00000000000000022e+00, 1.00000000000000022e+00])
    elif self.n == 3:
      return np.array([
          8.88888888888888840e-01, 5.55555555555555247e-01,
          5.55555555555555247e-01
      ])
    elif self.n == 4:
      return np.array([
          6.52145154862545984e-01, 6.52145154862545984e-01,
          3.47854845137453961e-01, 3.47854845137453961e-01
      ])
    elif self.n == 5:
      return np.array([
          5.68888888888888888e-01, 4.78628670499366471e-01,
          4.78628670499366471e-01, 2.36926885056189002e-01,
          2.36926885056189002e-01
      ])
    elif self.n == 6:
      return np.array([
          4.67913934572691037e-01, 4.67913934572691037e-01,
          3.60761573048138551e-01, 3.60761573048138551e-01,
          1.71324492379170384e-01, 1.71324492379170384e-01
      ])
    elif self.n == 7:
      return np.array([
          4.17959183673469403e-01, 3.81830050505118923e-01,
          3.81830050505118923e-01, 2.79705391489276589e-01,
          2.79705391489276589e-01, 1.29484966168869203e-01,
          1.29484966168869203e-01
      ])
    elif self.n == 8:
      return np.array([
          3.62683783378361990e-01, 3.62683783378361990e-01,
          3.13706645877887158e-01, 3.13706645877887158e-01,
          2.22381034453374482e-01, 2.22381034453374482e-01,
          1.01228536290376758e-01, 1.01228536290376758e-01
      ])
    elif self.n == 9:
      return np.array([
          3.30239355001259782e-01, 3.12347077040002696e-01,
          3.12347077040002696e-01, 2.60610696402935327e-01,
          2.60610696402935327e-01, 1.80648160694857507e-01,
          1.80648160694857507e-01, 8.12743883615745649e-02,
          8.12743883615745649e-02
      ])
    elif self.n == 10:
      return np.array([
          2.95524224714752926e-01, 2.95524224714752926e-01,
          2.69266719309996294e-01, 2.69266719309996294e-01,
          2.19086362515982069e-01, 2.19086362515982069e-01,
          1.49451349150580670e-01, 1.49451349150580670e-01,
          6.66713443086879298e-02, 6.66713443086879298e-02
      ])


class GaussLegendre2DTri(NumericalQuadrature):

  def points(self):
    if self.n == 1:
      return np.array([[1 / 3, 1 / 3]])
    elif self.n == 3:
      return np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
    elif self.n == 4:
      return np.array([[1 / 3, 1 / 3], [3 / 5, 1 / 5], [1 / 5, 3 / 5],
                       [1 / 5, 1 / 5]])
    elif self.n == 6:
      return np.array([[0.816847572980459, 0.091576213509771],
                       [0.091576213509771, 0.816847572980459],
                       [0.091576213509771, 0.091576213509771],
                       [0.108103018168070, 0.445948490915965],
                       [0.445948490915965, 0.108103018168070],
                       [0.445948490915965, 0.445948490915965]])
    elif self.n == 7:
      return np.array([[1 / 3, 1 / 3], [0.797426985353087, 0.101286507323456],
                       [0.101286507323456, 0.797426985353087],
                       [0.101286507323456, 0.101286507323456],
                       [0.470142064105115, 0.059715871789770],
                       [0.059715871789770, 0.470142064105115],
                       [0.470142064105115, 0.470142064105115]])
    elif self.n == 12:
      return np.array([[0.873821971016996, 0.063089014491502],
                       [0.063089014491502, 0.873821971016996],
                       [0.063089014491502, 0.063089014491502],
                       [0.501426509658179, 0.249286745170910],
                       [0.249286745170910, 0.501426509658179],
                       [0.249286745170910, 0.249286745170910],
                       [0.636502499121399, 0.310352451033785],
                       [0.310352451033785, 0.636502499121399],
                       [0.636502499121399, 0.053145049844816],
                       [0.310352451033785, 0.053145049844816],
                       [0.053145049844816, 0.310352451033785],
                       [0.053145049844816, 0.636502499121399]])
    elif self.n == 13:
      return np.array([[0.333333333333333, 0.333333333333333],
                       [0.479308067841920, 0.260345966079040],
                       [0.260345966079040, 0.479308067841920],
                       [0.260345966079040, 0.260345966079040],
                       [0.869739794195568, 0.065130102902216],
                       [0.065130102902216, 0.869739794195568],
                       [0.065130102902216, 0.065130102902216],
                       [0.048690315425316, 0.312865496004874],
                       [0.312865496004874, 0.048690315425316],
                       [0.638444188569810, 0.048690315425316],
                       [0.048690315425316, 0.638444188569810],
                       [0.312865496004874, 0.638444188569810],
                       [0.638444188569810, 0.312865496004874]])
    elif self.n == 16:
      return np.array([[0.333333333333333, 0.333333333333333],
                       [0.081414823414554, 0.459292588292723],
                       [0.459292588292723, 0.081414823414554],
                       [0.459292588292723, 0.459292588292723],
                       [0.658861384496480, 0.170569307751760],
                       [0.170569307751760, 0.658861384496480],
                       [0.170569307751760, 0.170569307751760],
                       [0.898905543365938, 0.050547228317031],
                       [0.050547228317031, 0.898905543365938],
                       [0.050547228317031, 0.050547228317031],
                       [0.008394777409958, 0.728492392955404],
                       [0.728492392955404, 0.008394777409958],
                       [0.263112829634638, 0.008394777409958],
                       [0.008394777409958, 0.263112829634638],
                       [0.263112829634638, 0.728492392955404],
                       [0.728492392955404, 0.263112829634638]])

  def weights(self):
    if self.n == 1:
      return np.array([1])
    elif self.n == 3:
      return np.array(
          [0.16666666666666666, 0.16666666666666666, 0.16666666666666666])
    elif self.n == 4:
      return np.array(
          [-0.5625, 0.520833333333333, 0.520833333333333, 0.520833333333333])
    elif self.n == 6:
      return np.array([
          0.109951743655322, 0.109951743655322, 0.109951743655322,
          0.223381589678011, 0.223381589678011, 0.223381589678011
      ])
    elif self.n == 7:
      return np.array([
          0.225000000000000, 0.125939180544827, 0.125939180544827,
          0.125939180544827, 0.132394152788506, 0.132394152788506,
          0.132394152788506
      ])
    elif self.n == 12:
      return np.array([
          0.050844906370207, 0.050844906370207, 0.050844906370207,
          0.116786275726379, 0.116786275726379, 0.116786275726379,
          0.082851075618374, 0.082851075618374, 0.082851075618374,
          0.082851075618374, 0.082851075618374, 0.082851075618374
      ])
    elif self.n == 13:
      return np.array([
          -0.149570044467682, 0.175615257433208, 0.175615257433208,
          0.175615257433208, 0.053347235608838, 0.053347235608838,
          0.053347235608838, 0.077113760890257, 0.077113760890257,
          0.077113760890257, 0.077113760890257, 0.077113760890257,
          0.077113760890257
      ])
    elif self.n == 16:
      return np.array([
          0.144315607677787, 0.095091634267285, 0.095091634267285,
          0.095091634267285, 0.103217370534718, 0.103217370534718,
          0.103217370534718, 0.032458497623198, 0.032458497623198,
          0.032458497623198, 0.027230314174435, 0.027230314174435,
          0.027230314174435, 0.027230314174435, 0.027230314174435,
          0.027230314174435
      ])


class GaussLegendre2DQuad(NumericalQuadrature):

  def points(self):
    if self.n == 1:
      return np.array([[-0.577735026, -0.577735026],
                       [0.577735026, -0.577735026], [0.577735026, 0.577735026],
                       [-0.577735026, 0.577735026]])
    else:
      raise NotImplementedError("Not implemented yet")

  def weights(self):
    if self.n == 1:
      return np.array([1, 1, 1, 1])
    else:
      raise NotImplementedError("Not implemented yet")


class GaussLegendre3DTri(NumericalQuadrature):

  def points(self):
    if self.n == 1:
      return np.array([[1 / 3, 1 / 3, 1 / 3]])
    elif self.n == 4:
      return np.array(
          [[0.5854101966249685, 0.2073433826145143, 0.2073433826145143],
           [0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
           [0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
           [0.1381966011250105, 0.5854101966249685, 0.1381966011250105]])
    elif self.n == 5:
      return np.array([[0.25, 0.25, 0.25], [0.5, 1 / 6, 1 / 6],
                       [1 / 6, 0.5, 1 / 6], [1 / 6, 1 / 6, 0.5],
                       [1 / 6, 1 / 6, 1 / 6]])
    else:
      raise NotImplementedError("Not implemented yet")

  def weights(self):
    if self.n == 1:
      return np.array([1])
    elif self.n == 4:
      return np.array([-0.8, 0.45, 0.45, 0.45])
    elif self.n == 5:
      return np.array([1 / 24, 9 / 24, 9 / 24, 9 / 24, 9 / 24])
    else:
      raise NotImplementedError("Not implemented yet")
