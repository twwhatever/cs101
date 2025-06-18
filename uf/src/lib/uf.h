#pragma once

#include <concepts>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Concept: T must be hashable and equality comparable
template <typename T>
concept HashEq = requires(T a, T b) {
  { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
  { a == b } -> std::convertible_to<bool>;
};

template <HashEq T> class UnionFind {
 public:
  // Add a new set containing only the given element.
  void add(const T& a) {
    if (id_.find(a) != id_.end()) {
      return;
    }
    size_t idx = parent_.size();
    id_[a] = idx;
    parent_.push_back(idx);
    size_from_root_[idx] = 1;
    ++set_count_;
  }

  // Connect the set containing a with the set containing b.
  void connect(const T& a, const T& b) {
    add(a);
    add(b);
    size_t root_a = find(a);
    size_t root_b = find(b);
    if (root_a == root_b) {
      return;
    }
    size_t size_a = size_from_root_[root_a];
    size_t size_b = size_from_root_[root_b];
    if (size_a < size_b) {
      parent_[root_a] = root_b;
      size_from_root_[root_b] += size_a;
      size_from_root_.erase(root_a);
    } else {
      parent_[root_b] = root_a;
      size_from_root_[root_a] += size_b;
      size_from_root_.erase(root_b);
    }
    --set_count_;
  }

  // Return true if a and b are in the same set.
  bool query(const T& a, const T& b) const {
    auto it_a = id_.find(a);
    auto it_b = id_.find(b);
    if (it_a == id_.end() || it_b == id_.end()) {
      return false;
    }
    size_t idx_a = it_a->second;
    size_t idx_b = it_b->second;
    // Find root without path compression
    while (parent_[idx_a] != idx_a) {
      idx_a = parent_[idx_a];
    }
    while (parent_[idx_b] != idx_b) {
      idx_b = parent_[idx_b];
    }
    return idx_a == idx_b;
  }

  // Return the size of set containing a.
  size_t set_count(const T& a) const {
    auto it = id_.find(a);
    if (it == id_.end()) {
      return 0;
    }
    size_t idx = it->second;
    // Find root without path compression
    while (parent_[idx] != idx) {
      idx = parent_[idx];
    }
    auto sz_it = size_from_root_.find(idx);
    if (sz_it == size_from_root_.end())
      return 0;
    return sz_it->second;
  }

  // Return the number of connected components.
  size_t size() const { return set_count_; }

 private:
  // Find the root for the given element.
  size_t find(const T& a) const {
    auto it = id_.find(a);
    if (it == id_.end()) {
      throw std::invalid_argument("Element not found");
    }
    size_t idx = it->second;
    size_t root = idx;
    // Find root
    while (parent_[root] != root) {
      root = parent_[root];
    }
    // Path compression
    while (parent_[idx] != root) {
      size_t next = parent_[idx];
      parent_[idx] = root;
      idx = next;
    }
    return root;
  }

  std::unordered_map<T, size_t> id_;
  mutable std::vector<size_t> parent_;
  std::unordered_map<size_t, size_t> size_from_root_;
  size_t set_count_{0};
};
