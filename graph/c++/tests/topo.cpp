#define BOOST_TEST_MODULE Topological sort tests
#include <boost/test/included/unit_test.hpp>

#include "../lib/graph.h"

using namespace std;

using namespace twwhatever::cs101::graph;

BOOST_AUTO_TEST_CASE(dfs_smoke_test)
{
  vector<pair<int, int>> graph {{1, 2}, {1, 3}, {2, 3}, {3, 4}, {3, 5}, {4, 5}};
  AdjListGraph<int> adj(graph.begin(), graph.end());
  PostOrderRecord<int> po;
  adj.dfs(po);
  auto expected = vector<int> {5, 4, 3, 2, 1};
  BOOST_TEST(po.postOrder == expected );
}
