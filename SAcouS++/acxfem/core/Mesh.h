#include <vector>
#include <array>
#include <memory>
#include <Python.h>


namespace SAcouS {
    namespace acxfem {
class MeshTriangle {
    private:
        std::vector<std::array<double, 3>> nodes;
        std::vector<std::array<int, 3>> elem_connectivity;
        std::vector<std::array<int, 3>> elem_faces;  // edge for 2D, face for 3D
    public:
        MeshTriangle();
        MeshTriangle(const std::vector<std::array<double, 3>>& nodes, const std::vector<std::array<int, 3>>& elem_connectivity, const std::vector<std::array<int, 3>>& elem_faces);
        ~MeshTriangle();
        void set_nodes(const std::vector<std::array<double, 3>>& nodes);
        void set_elem_connectivity(const std::vector<std::array<int, 3>>& elem_connectivity);
        void set_elem_faces(const std::vector<std::array<int, 3>>& elem_faces);
        std::vector<std::array<double, 3>> get_nodes() const;
        std::vector<std::array<int, 3>> get_elem_connectivity() const;
        std::vector<std::array<int, 3>> get_elem_faces() const;
        void read_mesh(const std::string& filename);
        };
    }  // namespace acxfem
}  // namespace SAcouS