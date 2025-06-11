#pragma once

#include <chrono>
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>

using namespace std;

// Cronometro
using Time = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Time>;

// Inicia el cronometro
inline TimePoint start_timer() {
    return Time::now();
}

// Detiene el cronometro y devuelve la duracion en segundos
inline double stop_timer(const TimePoint& start_time) {
    TimePoint end_time = Time::now();
    std::chrono::duration<double> duration = end_time - start_time;
    return duration.count();
}

// Imprime duracion en segundos con formato
inline void print_duration(double duration, const string& label) {
    cout << label << ": " << fixed << setprecision(2) << duration << " s" << endl;
}