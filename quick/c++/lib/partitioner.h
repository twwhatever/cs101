#pragma once

#include <iterator>

template<typename T> struct BasicPartitioner {
  void partition(T begin, T end, typename std::iterator_traits<T>::difference_type k) {
    std::swap(*(begin + k), *(end - 1));
    auto w = begin;
    
    for (auto it = begin; it != end - 1; ++it) {
      if (*it < *(end - 1)) {
        std::swap(*w++, *it);
      }
    }
    std::swap(*w, *(end - 1));
  }
};
    
