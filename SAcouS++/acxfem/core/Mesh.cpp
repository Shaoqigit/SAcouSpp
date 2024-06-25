#ifndef MESH_H
#define MESH_H

#include <vector>
#include <array>
#include <memory>
#include "Mesh.h"

// initialize the mesh for private members nodes and elem_connectivity
SAcouS::acxfem::MeshTriangle::MeshTriangle(const std::vector<std::array<double, 3>>& nodes, const std::vector<std::array<int, 3>>& elem_connectivity, const std::vector<std::array<int, 3>>& elem_faces): nodes(nodes), elem_connectivity(elem_connectivity), elem_faces(elem_faces) {}
    // Your code goes here


std::vector<std::array<double, 3>> SAcouS::acxfem::MeshTriangle::get_nodes() const {
    return nodes;
}

std::vector<std::array<int, 3>> SAcouS::acxfem::MeshTriangle::get_elem_connectivity() const {
    return elem_connectivity;
}

#endif // MESH_H