cmake_minimum_required(VERSION 3.25)
project(FFT_Project LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Настройка стандарта
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 2. Определение исполняемого файла
add_executable(fft_app 
    main.cpp
    fft_factory.cpp
    recursive_fft_aos.cpp
    recursive_fft_soa.cpp
    iterative_fft_soa.cpp
    types.cpp
)

# 3. Настройка специфичных для компиляторов опций
if(MSVC)
    # Настройки для Microsoft Visual C++
    target_compile_options(fft_app PRIVATE
        /fp:fast       # Включает агрессивную оптимизацию математики (в т.ч. finite-math)
        /permissive-   # Делает компилятор более строгим к стандарту
        /W4            # Высокий уровень предупреждений
        /Ox            # Максимальная оптимизация
        /arch:AVX2     # Включаем AVX2 инструкции
        /Zc:preprocessor # Включает стандартный препроцессор
        /Zc:__cplusplus  # Правильное значение макроса __cplusplus
    )
    target_compile_definitions(fft_app PRIVATE _USE_MATH_DEFINES NOMINMAX) # Для старых заголовков cmath
else()
    # Настройки для GCC и Clang
    target_compile_options(fft_app PRIVATE 
        -Wall -Wextra -O3
        -march=x86-64-v3    # AVX2, FMA, BMI2 (современный стандарт десктопов)
        
        # Настройки для агрессивной векторизации
        -fno-math-errno
        -fno-trapping-math
        -fno-signed-zeros
        -ffinite-math-only
        -fassociative-math   # Позволяет использовать FMA в FFT
    )

    # Если используется Clang, добавим специфичные флаги для векторизации
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(fft_app PRIVATE -Rpass=loop-vectorize)    
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Чтобы видеть, где векторизация прошла, а где нет (аналог -Rpass в Clang)
        target_compile_options(fft_app PRIVATE -fopt-info-vec-optimized)
    endif()
endif()

# 4. Настройка линковки (если потребуется математическая библиотека в Linux)
if(UNIX AND NOT APPLE)
    target_link_libraries(fft_app PRIVATE m)
endif()

# 5. Сообщение пользователю при генерации
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Standard C++: ${CMAKE_CXX_STANDARD}")