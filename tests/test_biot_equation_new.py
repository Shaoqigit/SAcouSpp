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

# Main Test case
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
working_dir = os.path.join(current_dir, "..")
import sys

sys.path.append(working_dir)

import numpy as np
import matplotlib.pyplot as plt

from SAcouS.Mesh import Mesh1D
from SAcouS.Materials import Air, PoroElasticMaterial
from SAcouS.PostProcess import PostProcessField

from analytical.Biot_sol import solve_PW

from SAcouS.acxfem import Helmholtz1DElement
from SAcouS.acxfem import FESpace
from SAcouS.acxfem import BiotAssembler
from SAcouS.acxfem import ApplyBoundaryConditions
from SAcouS.acxfem import check_material_compability
from SAcouS.acxfem import LinearSolver


def test_case():
  # ====================== Pysical Problem ======================
  # define the materials
  # given JCA porous material properties
  phi = 0.99    # porosity
  sigma = 1.0567e4    # resistivity
  alpha = 1.2    # Tortuosity
  Lambda_prime = 490e-6    # Viscous characteristic length
  Lambda = 240e-6    #
  rho_1 = 9.2
  nu = 0.285
  E = 3.155e5
  eta = 0.032
  xfm = PoroElasticMaterial('xfm', phi, sigma, alpha, Lambda_prime, Lambda,
                            rho_1, E, nu, eta)

  # Harmonic Acoustic problem define the frequency
  freq = 2000
  omega = 2 * np.pi * freq    # angular frequency
  xfm.set_frequency(omega)
  k_0 = omega / Air.c
  theta = 0    #indidence angle
  ky = k_0 * np.sin(theta * np.pi / 180)
  kx = k_0 * np.cos(theta * np.pi / 180)

  num_elem = 500    # number of elements
  num_nodes = num_elem + 1    # number of nodes

  nodes = np.linspace(-1, 0, num_nodes)
  # print()
  elem_connec1 = np.arange(0, num_elem)
  elem_connec2 = np.arange(1, num_nodes)
  connectivity = np.vstack((elem_connec1, elem_connec2)).T
  # print(connectivity)

  # read the mesh data structure
  mesh = Mesh1D(nodes, connectivity)
  # mesh.refine_mesh(1)

  elements2node = mesh.get_mesh(
  )    # dict: elements number with nodes coodinates
  # define the subdomains: domain name (material) and the elements in the domain
  xfm_elements = np.arange(0, num_elem)
  subdomains = {xfm: xfm_elements}
  check_material_compability(subdomains)

  order = 3    # global order of the bases
  # applied the basis on each element
  for mat, elems in subdomains.items():
    if mat.TYPE == 'Poroelastic':
      Pb_bases = [
          Helmholtz1DElement('Pb', order, elements2node[elem],
                             (1 / mat.rho_f, 1 / mat.K_f, mat.gamma_til))
          for elem in elems
      ]    # basis for pressure in porous domain
      Ux_bases = [
          Helmholtz1DElement('Ux', order, elements2node[elem],
                             (mat.P_hat, mat.rho_til, mat.gamma_til))
          for elem in elems
      ]    #

  # handler the dofs: map the basis to mesh
  fe_space = FESpace(mesh, subdomains, Pb_bases, Ux_bases)

  # initialize the assembler
  Biot_assember = BiotAssembler(fe_space, dtype=np.complex128)
  Biot_assember.assembly_global_matrix([Pb_bases, Ux_bases], ['Pb', 'Ux'])
  left_hand_matrix = Biot_assember.get_global_matrix(omega)

  right_hand_vec = np.zeros(Biot_assember.nb_global_dofs, dtype=np.complex128)
  # import pdb;pdb.set_trace()

  # ============================= Boundary conditions =====================================
  BCs_applier = ApplyBoundaryConditions(mesh, fe_space, left_hand_matrix,
                                        right_hand_vec, omega)

  essential_bcs = {
      'type': 'solid_displacement',
      'value': 0,
      'position': 0.
  }    # position is the x coordinate
  BCs_applier.apply_essential_bc(essential_bcs, var='Ux', bctype='nitsche')
  #  natural boundary condition
  nature_bcs = {
      'type': 'total_displacement',
      'value': 1,
      'position': -1.
  }    # position is the x coordinate
  BCs_applier.apply_nature_bc(nature_bcs, var='Pb')

  # ============================= Solve the linear system ================================
  # solver the linear system
  linear_solver = LinearSolver(fe_space=fe_space)
  # print("condition number:", linear_solver.condition_number(left_hand_matrix))
  # plot_matrix_partten(left_hand_matrix)
  linear_solver.solve(BCs_applier.left_hand_side, BCs_applier.right_hand_side)
  sol = linear_solver.u

  # ================================== Analytical Solution ======================
  # analytical solution
  xfm.set_frequency(omega)
  ana_sol = solve_PW(xfm, ky, nodes, 1)
  # plot the solution
  post_processer_p = PostProcessField(mesh.nodes,
                                      r'1D Biot (2000$Hz$) Pressure',
                                      quantity='Pressure',
                                      unit='Pa')
  post_processer_p.plot_sol(
      (np.real(sol[:num_elem + 1]), f'FEM ($p=3$)', 'solid'),
      (np.real(ana_sol[4, :]), 'Analytical', 'dashed'))
  # post_processer.plot_sol((np.real(sol[:101]), f'FEM ($p=3$)', 'solid'))
  plt.show(block=False)
  plt.pause(1)

  post_processer_u = PostProcessField(mesh.nodes,
                                      r'1D Biot (2000$Hz$) Solid displacement',
                                      quantity='Displacement',
                                      unit='m')
  post_processer_u.plot_sol(
      (np.real(sol[num_elem + 1:]), f'FEM ($p=3$)', 'solid'),
      (np.real(ana_sol[1, :]), 'Analytical', 'dashed'))
  # post_processer.plot_sol((np.real(sol[:101]), f'FEM ($p=3$)', 'solid'))test
  plt.show(block=False)
  # plt.show()
  plt.pause(1)
  plt.close('all')

  error_p = post_processer_p.compute_error(sol[:num_elem + 1], ana_sol[4, :])
  error_u = post_processer_u.compute_error(sol[num_elem + 1:], ana_sol[1, :],
                                           -1)

  print("error_p: ", error_p)
  print("error_u: ", error_u)
  if error_p < 1e-4 and error_u < 1e-3:
    print("Test passed!")
    return True
  else:
    print("Test failed!")
    return False


if __name__ == "__main__":
  result = test_case()
