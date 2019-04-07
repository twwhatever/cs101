#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace twwhatever::cs101::graph {

  template<typename T> class AdjListGraph {
    std::unordered_map<T, std::unordered_set<T>> adj_;

    template<typename U> class DfsHelper {
    private:
      U& callbacks_;

      // bookeeping
      std::unordered_set<T> discovered_;
      std::unordered_set<T> processed_;

    public:
      DfsHelper(U& callback) : callbacks_(callback) { }

      bool discovered(const T& node) { return discovered_.count(node) > 0; }

      bool processed(const T& node) { return processed_.count(node) > 0; }

      void discover(const T& node) {
        discovered_.insert(node);
        callbacks_.discover(node);
      }

      void process(const T& node) {
        processed_.insert(node);
        callbacks_.process(node);
      }

      void edge(const T& source, const T& dest) {
        callbacks_.edge(source, dest);
      }

      bool finished() {
        return callbacks_.finished();
      }
    };

    // runs DFS through the whole graph
    template<typename U> void dfs_(DfsHelper<U>& userRecord) {
      for (const auto& edges : adj_) {
        if (userRecord.processed(edges.first)) { continue; }
        dfs_(userRecord, edges.first);
      }
    }

    // rooted dfs starting at the vertex indicated by start
    // will not traverse nodes that are not reachable from start
    template<typename U> void dfs_(DfsHelper<U>& userRecord, const T& start) {
      if (userRecord.finished()) { return; }
      userRecord.discover(start);

      for (const auto& node : adj_[start]) {
        if (!userRecord.discovered(node)) {
          userRecord.edge(start, node);
          dfs_(userRecord, node);
        }
      }

      userRecord.process(start);
    }

  public:
    template<typename Iter> AdjListGraph(Iter begin, Iter end) {
      for (auto it = begin; it != end; ++it) {
        adj_[std::get<0>(*it)].insert(std::get<1>(*it));
      }
    }

    template<typename U> void dfs(U& u) {
      DfsHelper<U> helper(u);
      dfs_(helper);
    }

    template<typename U> void dfs(U& u, const T& start) {
      DfsHelper<U> helper(u);
      dfs_(helper);
    }
  };

  template<typename T> struct PostOrderRecord {
    std::vector<T> postOrder;

    void discover(const T& node) { }

    void process(const T& node) {
      postOrder.push_back(node);
    }

    void edge(const T&, const T&) { }

    bool finished() { return false; }
  };
}
