cmake_minimum_required(VERSION 3.25)
project(FFT_Project LANGUAGES CXX)

# 1. Настройка стандарта C++23
# Требуется для std::ranges::to, std::views::zip и std::print
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 2. Определение исполняемого файла
add_executable(fft_app 
    main.cpp 
    iterative_fft.cpp 
    recursive_fft.cpp
)

# 3. Настройка специфичных для компиляторов опций
if(MSVC)
    # Настройки для Microsoft Visual C++
    target_compile_options(fft_app PRIVATE 
        /W4            # Высокий уровень предупреждений
        /Ox            # Максимальная оптимизация
        /arch:AVX2     # Включаем AVX2 инструкции
        /Zc:preprocessor # Корректный препроцессор для C++23
        /Zc:__cplusplus  # Правильное значение макроса __cplusplus
        /utf-8         # Поддержка UTF-8 в исходниках (для кириллицы в std::print)
    )
    target_compile_definitions(fft_app PRIVATE _USE_MATH_DEFINES) # Для старых заголовков cmath
else()
    # Настройки для GCC и Clang
    target_compile_options(fft_app PRIVATE 
        -Wall 
        -Wextra 
        -O3 
        # x86-64-v3 включает: SSE4.2, AVX, AVX2, BMI2, FMA. 
        # Это идеально подходит для ваших SIMD-ядер.
        -march=x86-64-v3 
        # Отключаем установку errno для мат. функций (ускоряет std::polar/sin/cos)
        -fno-math-errno
    )

    # Если используется Clang, добавим специфичные флаги для векторизации
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(fft_app PRIVATE -Rpass=loop-vectorize)
    endif()
endif()

# 4. Настройка линковки (если потребуется математическая библиотека в Linux)
if(UNIX AND NOT APPLE)
    target_link_libraries(fft_app PRIVATE m)
endif()

# 5. Сообщение пользователю при генерации
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Standard C++: ${CMAKE_CXX_STANDARD}")
