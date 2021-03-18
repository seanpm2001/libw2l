cmake_minimum_required(VERSION 3.5.1)
project(libw2l)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_definitions(-flto=thin)
find_package(GFLAGS REQUIRED)
find_library(LIBOMP NAMES libomp.a)
find_package(Glog REQUIRED)
find_package(Threads REQUIRED)
find_package(ArrayFire REQUIRED)

add_library(w2l SHARED
    src/w2l_encode.cpp
    src/w2l_encode_backend.cpp
)

if (W2L_USE_MKL)
    target_compile_definitions(w2l PUBLIC -DUSE_MKL -DW2L_LIBRARIES_USE_MKL)
    find_library(MKL_CORE NAMES libmkl_core.a)
    find_library(MKL_INTEL_LP64 NAMES libmkl_intel_lp64.a)
    find_library(MKL_GNU_THREAD NAMES libmkl_gnu_thread.a)
    target_link_libraries(w2l "-Wl,--start-group" ${MKL_CORE} ${MKL_INTEL_LP64} ${MKL_GNU_THREAD} "-Wl,--end-group")
endif()

if (W2L_USE_ACCELERATE)
    target_link_libraries(w2l "-framework Accelerate")
else()
    find_library(LIBIPPS NAMES libipps.a HINTS /opt/intel/ipp/lib)
    find_library(LIBIPPVM NAMES libippvm.a HINTS /opt/intel/ipp/lib)
    find_library(LIBIPPCORE NAMES libippcore.a HINTS /opt/intel/ipp/lib)
    set(IPP_LIBS ${LIBIPPS} ${LIBIPPCORE} ${LIBIPPVM})
    target_link_libraries(w2l ${IPP_LIBS})
endif()

set_target_properties(ArrayFire::afcpu PROPERTIES INTERFACE_LINK_LIBRARIES "")
target_link_libraries(w2l
    "-L${PROJECT_SOURCE_DIR}/../opt/lib"
    afcpu
    dnnl
    flashlight
    flashlight-app-asr
    fl-libraries
    sndfile # FIXME
    ${GFLAGS_LIBRARIES}
    glog
    ${LIBOMP}
)

# openmp
if (APPLE)
    target_link_libraries(w2l -Wl,-cache_path_lto,/tmp/lto-cache)
    target_compile_options(w2l PUBLIC -Xpreprocessor -fopenmp)
elseif (CMAKE_SYSTEM_NAME MATCHES "Windows")
    # no openmp on windows yet
else()
    target_compile_options(w2l PUBLIC -fopenmp)
endif()

target_include_directories(
  w2l
  PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/../opt/include/cereal
)
install(TARGETS w2l DESTINATION lib)
install(FILES src/w2l_common.h src/w2l_encode.h DESTINATION include)

add_executable(w2l_cli src/cli.cpp)
target_link_libraries(w2l_cli w2l dl)
install(TARGETS w2l_cli DESTINATION bin)
