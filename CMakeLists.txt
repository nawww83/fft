cmake_minimum_required(VERSION 3.25) # Версия с хорошей поддержкой C++20/23
project(FFT_Project LANGUAGES CXX)

# Требуется C++23 для std::ranges::to, std::views::zip и std::print
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fft_app main.cpp iterative_fft.cpp recursive_fft.cpp)

# Настройка оптимизаций
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Обработка Clang (включая clang-cl на Windows)
    target_compile_options(fft_app PRIVATE -Wall -O3)

elseif(MSVC)
    # Только "чистый" MSVC (cl.exe)
    # /permissive- делает компилятор более строгим к стандарту (нужно для C++20/23)
    # /Zc:__cplusplus заставляет макрос __cplusplus выдавать верную дату
    target_compile_options(fft_app PRIVATE /W4 /Ox /arch:AVX2 /permissive- /Zc:__cplusplus)
else()
    # GCC / Другие
    # -march=x86-64-v2 даст базовое ускорение (SSE4.2), 
    # но не сломает совместимость с большинством ПК.
    target_compile_options(fft_app PRIVATE -Wall -Wextra -O3 -march=x86-64-v2)
endif()


# Если используете старый Clang/GCC, может потребоваться линковка с библиотекой для std::print
# или использование fmtlib, но в актуальных версиях (GCC 14+, Clang 18+) работает из коробки.
