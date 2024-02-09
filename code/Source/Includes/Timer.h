/*
 * @file Timer.h
 * @author https://www.learncpp.com/cpp-tutorial/timing-your-code/
 * @version 1.0 30.12.2020
 * */

#ifndef SIMULATION_TIMER_H
#define SIMULATION_TIMER_H
#include <chrono>

class Timer {
private:
    // Type aliases to make accessing nested type easier
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<double, std::ratio<1> >;

    std::chrono::time_point<clock_t> m_beg;

public:
    Timer() : m_beg(clock_t::now())
    {
    }

    void reset()
    {
        m_beg = clock_t::now();
    }

    [[nodiscard]] double elapsed() const
    {
        return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count();
    }


};


#endif //SIMULATION_TIMER_H
