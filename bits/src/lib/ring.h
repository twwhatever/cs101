#pragma once

#include <array>
#include <stdexcept>

template<typename T, std::size_t N> class StaticRingBuffer {
 public:
  
  void enqueue(const T& elem) {
    buffer_.at(tail_) = elem;
    inc(tail_);

    if (tail_ == head_) {
      inc(head_);
    }
  }

  T dequeue() {
    if (empty()) {
      throw std::underflow_error("empty queue");
    }

    auto val = buffer_.at(head_);
    inc(head_);
    return val;
  }

  size_t size() const {
    return (tail_ + (N + 1) - head_) % (N + 1);
  }

  bool empty() const {
    return head_ == tail_;
  }

 private:

  void inc(size_t& val) {
    val = (val + 1) % (N + 1);
  }

  std::size_t head_{0};
  std::size_t tail_{0};
  std::array<T, N + 1> buffer_; 
};