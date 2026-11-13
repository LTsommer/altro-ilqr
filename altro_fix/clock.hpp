//
// Created by LTsommer on 2024/5/31.
//

#ifndef CILQR_CLOCK_HPP
#define CILQR_CLOCK_HPP

#include <chrono>
#include <cstdint>

class StopWatch {
public:
    StopWatch();

    void start();

    uint64_t elapsed_milliseconds();

    uint64_t elapsed_microseconds();

private:
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
};

#endif //CILQR_CLOCK_HPP
