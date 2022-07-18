#ifndef LOGGING_HPP_
#define LOGGING_HPP_

#include <iostream>
#include <chrono>

#define LOG_TO_STREAM(OSTREAM, TAG) (OSTREAM) << '[' << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() << "] " << (#TAG) << ": "
#define LOG(TAG) LOG_TO_STREAM(std::cout, TAG)

#endif // LOGGING_HPP_