#include <vector>

struct EdgeGraph {
  std::vector<int> edge_src;
  std::vector<int> edge_dst;
  std::vector<float> edge_rbf;

  std::size_t num_edges() const { return edge_src.size(); }
};
