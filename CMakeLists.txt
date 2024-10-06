cmake_minimum_required(VERSION 3.10)

# Nome del progetto
project(FirstChallengeLIS)

# Imposta lo standard C++ a C++11
set(CMAKE_CXX_STANDARD 11)

# Aggiungi il file sorgente
add_executable(first_challenge_lis first_challenge_lis.cpp)

# Trova i pacchetti richiesti
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LIS REQUIRED)

# Includi le directory per i pacchetti trovati
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${LIS_INCLUDE_DIRS})

# Linka le librerie richieste
target_link_libraries(challenge_one ${OpenCV_LIBS} ${LIS_LIBRARIES})