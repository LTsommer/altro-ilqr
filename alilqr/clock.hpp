//
// Created by LTsommer on 2024/5/31.
//

#ifndef CILQR_CLOCK_HPP
#define CILQR_CLOCK_HPP

#include <chrono>

class StopWatch {
public:
    StopWatch() { start(); }

    void start() { start_time_ = std::chrono::steady_clock::now(); }

    uint64_t elapsed_milliseconds() {
        const auto end_time = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                     start_time_)
                .count();
    }

    uint64_t elapsed_microseconds() {
        const auto end_time = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     start_time_)
                .count();
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
};

#endif //CILQR_CLOCK_HPP
