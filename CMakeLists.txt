cmake_minimum_required(VERSION 3.16)
project(vane LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(vane_core
  src/core.cpp src/pricer.cpp
  src/market.cpp src/flow.cpp src/desk.cpp src/policy.cpp src/simulator.cpp)
target_include_directories(vane_core PUBLIC include)
target_compile_options(vane_core PRIVATE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)

add_executable(vane-quote tools/vane_quote.cpp)
target_link_libraries(vane-quote PRIVATE vane_core)

add_executable(vane-sim tools/vane_sim.cpp)
target_link_libraries(vane-sim PRIVATE vane_core)

add_executable(vane-backtest tools/vane_backtest.cpp)
target_link_libraries(vane-backtest PRIVATE vane_core)

enable_testing()

add_executable(test_unit tests/test_unit.cpp)
target_include_directories(test_unit PRIVATE tests)
target_link_libraries(test_unit PRIVATE vane_core)
add_test(NAME unit COMMAND test_unit)

add_executable(test_property tests/test_property.cpp)
target_include_directories(test_property PRIVATE tests)
target_link_libraries(test_property PRIVATE vane_core)
add_test(NAME property COMMAND test_property)

add_executable(test_sim tests/test_sim.cpp)
target_include_directories(test_sim PRIVATE tests)
target_link_libraries(test_sim PRIVATE vane_core)
add_test(NAME simulation COMMAND test_sim)

add_executable(test_policy tests/test_policy.cpp)
target_include_directories(test_policy PRIVATE tests)
target_link_libraries(test_policy PRIVATE vane_core)
add_test(NAME policy COMMAND test_policy)
