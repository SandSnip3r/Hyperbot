#ifndef LOGGING_HPP_
#define LOGGING_HPP_

#include <absl/log/log.h>

// #include <iostream>
// #include <chrono>

// #define LOG_TO_STREAM(OSTREAM) (OSTREAM) << '[' << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() << "] " << (__FUNCTION__) << ':' << (__LINE__) << ": "
// #define HYPERBOT_LOG() LOG_TO_STREAM(std::cout)
#define HYPERBOT_LOG() LOG(INFO)

#endif // LOGGING_HPP_