#include "clock.hpp"

StopWatch::StopWatch() {
    start();
}

void StopWatch::start() {
    start_time_ = std::chrono::steady_clock::now();
}

uint64_t StopWatch::elapsed_milliseconds() {
    const auto end_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
}

uint64_t StopWatch::elapsed_microseconds() {
    const auto end_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count();
}
