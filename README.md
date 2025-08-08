# cuda_python

# CUDA 학습자를 위한 ufunc 완벽 가이드

이제 막 CUDA 공부를 시작하셨고 'ufunc'라는 개념을 접하셨다면, 고성능 병렬 컴퓨팅의 핵심에 한 걸음 더 다가선 것입니다. 이 문서에서는 ufunc이 무엇인지, 그리고 CUDA 프로그래밍에서 어떻게 강력한 도구로 사용되는지에 대해 자세히 설명합니다.

## 1. ufunc (Universal Function) 이란?

Ufunc은 "Universal Function"의 약자로, NumPy 라이브러리의 핵심 기능입니다. [15] 간단히 말해, 배열(Array)의 모든 요소에 대해 동일한 연산을 개별적으로, 그리고 매우 빠르게 수행하는 함수를 의미합니다. [16]

예를 들어, 100만 개의 숫자가 담긴 배열의 모든 숫자에 1을 더해야 하는 상황을 가정해 봅시다. 파이썬의 반복문(for loop)을 사용한다면 100만 번의 반복이 필요하지만, NumPy의 ufunc을 사용하면 단 한 줄의 코드로 이 작업을 훨씬 빠르게 처리할 수 있습니다. 이러한 방식을 **벡터화(Vectorization)**라고 부릅니다. [17]

**ufunc의 주요 특징:**

*   **요소별(Element-wise) 연산:** 배열의 각 요소에 대해 독립적으로 연산을 수행합니다.
*   **고속 연산:** 내부적으로 최적화된 컴파일된 C 코드로 구현되어 있어 순수 파이썬 코드보다 월등히 빠릅니다.
*   **브로드캐스팅(Broadcasting):** 모양(shape)이 다른 배열 간에도 특정 규칙에 따라 연산을 가능하게 합니다. [15]
*   **다양한 함수 제공:** `np.add`, `np.subtract`와 같은 기본적인 산술 연산부터 `np.sin`, `np.log`와 같은 복잡한 수학 함수까지 광범위한 ufunc이 내장되어 있습니다. [15]

## 2. CUDA와 ufunc의 만남: Numba

그렇다면 ufunc이 어떻게 GPU를 활용하는 CUDA와 연결될까요? 그 해답은 **Numba**라는 파이썬 JIT(Just-In-Time) 컴파일러에 있습니다. [3]

Numba는 파이썬 함수에 특정 데코레이터(`@decorator`)를 붙이는 것만으로 해당 함수를 GPU에서 병렬로 실행되는 코드로 변환해주는 놀라운 도구입니다. [1] 특히, Numba는 개발자가 직접 CUDA 코드를 작성하는 복잡한 과정 없이, 파이썬으로 작성된 간단한 스칼라(scalar) 함수를 GPU에서 동작하는 ufunc으로 만들 수 있게 지원합니다. [3]

Numba가 자동으로 처리해주는 작업들은 다음과 같습니다: [3]

*   입력된 모든 요소에 대해 병렬로 연산을 수행하는 CUDA 커널(kernel) 컴파일
*   입력과 출력을 위한 GPU 메모리 할당
*   CPU 메모리(Host)에서 GPU 메모리(Device)로 데이터 복사
*   데이터 크기에 맞춰 적절한 차원으로 CUDA 커널 실행
*   결과를 다시 CPU 메모리로 복사하여 반환

## 3. Numba로 CUDA ufunc 만들기

Numba는 CUDA ufunc을 생성하기 위해 মূলত 두 가지 데코레이터를 제공합니다: `@vectorize`와 `@guvectorize`.

### 3.1. `@vectorize`: 간단한 요소별 ufunc 생성

`@vectorize`는 스칼라 값을 입력받아 스칼라 값을 반환하는 간단한 함수를 GPU용 ufunc으로 만들어줍니다. [8]

**사용법:**

`@vectorize` 데코레이터에 함수의 입력 및 반환 값의 타입을 명시하고, `target='cuda'` 옵션을 설정합니다.

```python
import numpy as np
from numba import vectorize

# 각 요소에 2를 곱하고 1을 더하는 CUDA ufunc 생성
@vectorize(['float32(float32)'], target='cuda')
def add_two_and_one(x):
  return x * 2 + 1

# 데이터 준비
data = np.arange(10, dtype=np.float32)

# CUDA ufunc 실행
result = add_two_and_one(data)

print(result)
# 출력: [ 1.  3.  5.  7.  9. 11. 13. 15. 17. 19.]
<img width="538" height="395" alt="image" src="https://github.com/user-attachments/assets/3a6ed6ae-2243-4fdf-89c8-18097381873c" />
<img width="763" height="552" alt="image" src="https://github.com/user-attachments/assets/10a152f8-e599-478e-b951-740b0590600d" />
<img width="766" height="430" alt="image" src="https://github.com/user-attachments/assets/f8ae8f92-58de-45ca-b583-5d862c1e0464" />
더 많은 센서들을 특히 영상 제어에 병렬 처리는 무척 중요하다.

<img width="768" height="543" alt="image" src="https://github.com/user-attachments/assets/75c39159-dedd-4c3c-b168-bf9e7f4de4c7" />

<img width="762" height="552" alt="image" src="https://github.com/user-attachments/assets/d1a28cb0-6cb1-458d-af80-0883d54a7780" />
<img width="762" height="427" alt="image" src="https://github.com/user-attachments/assets/a7949f3a-7857-4a5b-be7f-27ef5de074d7" />

## See: https://github.com/googlecolab/colabtools/issues/5081#issuecomment-2629611179
!uv pip install -q --system numba-cuda==0.4.0
from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
import os
os.environ['NUMBA_CUDA_ARCH'] = 'sm_75'
import numba
import numpy as np
from numba import vectorize, jit, cuda
print("Numpy version: ", np.__version__)
print("numba version: ", numba.__version__)
print("Cuda avilable") if cuda.detect() else print("Cuda not avilable")
# Day 5: CUDA 파이썬을 이용한 자율주행 데이터 가속

**과정 목표:** 지난 4일간 배운 자율주행 데이터 처리 알고리즘들을 GPU를 이용해 가속하는 방법을 배웁니다. Numba 라이브러리를 사용하여 Python으로 직접 CUDA 코드를 작성하고, 병렬 처리의 핵심 원리와 최적화 기법을 실습을 통해 익힙니다.
---
## Lab 1: Numba Ufunc, 프로파일링, 그리고 정밀도

**실습 목표:**
1. 간단한 이미지 처리 작업을 위한 GPU 범용 함수(ufunc)를 작성합니다.
2. 데이터 크기에 따른 CPU와 GPU의 성능을 `%%timeit`으로 비교하여 메모리 전송 오버헤드를 확인합니다.
3. `float32`와 `float64`의 정밀도 차이가 계산 결과에 미치는 영향을 직접 확인합니다.
### 준비: 라이브러리 임포트 및 데이터 생성
import numpy as np
from numba import vectorize, jit
import time

# 자율주행 카메라 이미지를 모방한 가상 데이터 생성
def create_image(size_mb):
    # 1 pixel = 1 byte (uint8)
    num_pixels = size_mb * 1024 * 1024
    # 이미지의 가로:세로 비율을 16:9로 가정
    height = int(np.sqrt(num_pixels * 9 / 16))
    width = int(height * 16 / 9)
    print(f"생성된 이미지 크기: {width}x{height} ({width*height/1024/1024:.2f} MB)")
    return np.random.randint(0, 256, size=(height, width), dtype=np.uint8)

small_image = create_image(1)    # 약 1MB 크기 이미지
large_image = create_image(100)  # 약 100MB 크기 이미지
### 파트 1: 성능 비교 - 이미지 임계값 처리

이미지에서 특정 밝기 값(임계값)보다 밝은 픽셀은 흰색(255)으로, 어두운 픽셀은 검은색(0)으로 만드는 것은 객체 감지의 가장 기본적인 전처리 단계입니다.
# TODO: 아래 함수를 Numba의 @vectorize 데코레이터를 사용하여 GPU ufunc으로 변환하세요.
# target을 'cuda'로 설정하는 것을 잊지 마세요.
# 타입 시그니처: uint8를 입력받아 uint8를 반환 -> ['uint8(uint8, uint8)']

from numba import vectorize, uint8, cuda

@vectorize([uint8(uint8, uint8)], target='cuda')
def gpu_threshold(pixel, threshold):
    return 255 if pixel > threshold else 0

# CPU 버전 (NumPy)
def cpu_threshold_numpy(image, threshold):
    return np.where(image > threshold, 255, 0).astype(np.uint8)
#### 작은 이미지 성능 측정
print("--- 작은 이미지 (1MB) 성능 비교 ---")
print("CPU (NumPy):")
%timeit cpu_threshold_numpy(small_image, 128)

print("\nGPU (Numba ufunc):")
# 첫 실행은 컴파일 시간 포함
_ = gpu_threshold(small_image, 128)
%timeit gpu_threshold(small_image, 128)


#### 큰 이미지 성능 측정
print("--- 큰 이미지 (100MB) 성능 비교 ---")
print("CPU (NumPy):")
%timeit cpu_threshold_numpy(large_image, 128)

print("\nGPU (Numba ufunc):")
_ = gpu_threshold(large_image, 128)
%timeit gpu_threshold(large_image, 128)
#### 분석 질문
1. 작은 이미지와 큰 이미지에서 CPU와 GPU의 성능 차이는 어떻게 나타났나요?
2. 왜 이런 차이가 발생했을까요? 강의에서 배운 '메모리 전송 오버헤드' 개념과 연관지어 설명해보세요.
### 파트 2: 정밀도 문제 (Nefarious Example)

강의에서 본 '치명적 상쇄(Catastrophic Cancellation)' 문제를 직접 확인해봅시다.
import numpy as np
from numba import vectorize

# Corrected version to demonstrate precision loss
@vectorize(['float32(float32)'], target='cuda')
def precision_test_f32_corrected(x):
    # Force the literal '1.0' to be a 32-bit float
    one_f32 = np.float32(1.0)
    return (one_f32 + x) - one_f32

@vectorize(['float64(float64)'], target='cuda')
def precision_test_f64(x):
    # For float64, using a Python literal is fine as it's already 64-bit
    return (1.0 + x) - 1.0

# Using a value that WILL be lost in float32 but not float64
# float32 machine epsilon is ~1.19e-7. 1e-8 is smaller than that.
val = 1e-8
x_f32 = np.array([val], dtype=np.float32)
x_f64 = np.array([val], dtype=np.float64)

# Run the corrected f32 version and the f64 version
result_f32 = precision_test_f32_corrected(x_f32)
result_f64 = precision_test_f64(x_f64)

print(f"입력 값: {val}")
print(f"Corrected Float32 결과: {result_f32[0]}")
print(f"Float64 결과: {result_f64[0]}")
#### 분석 질문
1. `float32`와 `float64`의 결과가 왜 다르게 나왔나요?
2. 만약 Day 1의 칼만 필터나 Day 4의 PID 제어기처럼 정밀한 계산이 필요한 알고리즘을 GPU로 가속한다면, 이 결과가 어떤 중요한 점을 시사할까요?
---
## Lab 2: 커스텀 커널과 메모리 Coalescing

**실습 목표:**
1. 2D 이미지(행렬) 처리를 위한 커스텀 CUDA 커널을 작성합니다.
2. 메모리 접근 패턴(Coalesced vs. Uncoalesced)이 성능에 미치는 극적인 영향을 `%%timeit`과 프로파일러로 직접 확인하고 분석합니다.
### 준비: 라이브러리 임포트 및 데이터 생성
from numba import cuda

# 2048x2048 크기의 가상 행렬 데이터 생성
N = 2048
matrix = np.arange(N * N, dtype=np.float32).reshape(N, N)

# GPU로 데이터 전송
d_matrix = cuda.to_device(matrix)
d_transposed = cuda.device_array_like(d_matrix)
### 파트 1: 행렬 전치(Transpose) 커널 작성

행렬 전치는 메모리 접근 패턴의 중요성을 보여주는 고전적인 예제입니다.
@cuda.jit
def transpose_uncoalesced_kernel(A, B):
    # 2D 그리드에서 스레드의 x, y 좌표를 얻습니다.
    x, y = cuda.grid(2)

    # 경계 검사
    if x < B.shape[0] and y < B.shape[1]:
        # TODO: Uncoalesced 접근을 유발하는 코드를 작성하세요.
        # 힌트: 인접 스레드(x가 1씩 변함)가 메모리 상에서 멀리 떨어진 위치에 쓰도록 만드세요.
        # B[y, x] = A[x, y] 와 같은 형태가 될 것입니다.
        B[y, x] = A[x, y]

@cuda.jit
def transpose_coalesced_kernel(A, B):
    x, y = cuda.grid(2)

    if x < B.shape[0] and y < B.shape[1]:
        # TODO: Coalesced 접근이 일어나도록 코드를 작성하세요.
        # 힌트: 인접 스레드가 메모리 상에서 연속된 위치에 쓰도록 만드세요.
        # B[x, y] = A[y, x] 와 같은 형태가 될 것입니다.
        B[x, y] = A[y, x]
### 파트 2: 성능 측정 및 프로파일링

두 커널의 실행 시간을 측정하고 비교해봅시다.
# 실행 구성 설정
threads_per_block = (32, 32)
blocks_per_grid_x = int(np.ceil(matrix.shape[0] / threads_per_block[0]))
blocks_per_grid_y = int(np.ceil(matrix.shape[1] / threads_per_block[1]))
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

print("--- Uncoalesced Kernel 성능 ---")
%timeit transpose_uncoalesced_kernel[blocks_per_grid, threads_per_block](d_matrix, d_transposed)

print("\n--- Coalesced Kernel 성능 ---")
%timeit transpose_coalesced_kernel[blocks_per_grid, threads_per_block](d_matrix, d_transposed)
#### (선택) 프로파일러로 확인하기

만약 로컬 환경에 CUDA Toolkit이 설치되어 있다면, 위 커널들을 별도의 `.py` 파일로 저장한 뒤 터미널에서 아래 명령어를 실행하여 메모리 처리량을 직접 확인할 수 있습니다.

`$ nsys profile python your_script.py`
#### 분석 질문
1. 두 커널의 성능 차이가 얼마나 컸나요?
2. 왜 이런 차이가 발생했는지 '메모리 병합(Coalescing)'과 '행 우선 저장(Row-major Layout)' 개념을 사용하여 설명하세요.
---
## Lab 3: 원자적 연산과 경쟁 상태

**실습 목표:**
1. 여러 스레드가 공유 자원에 동시에 접근할 때 발생하는 경쟁 상태(Race Condition) 문제를 직접 재현합니다.
2. 원자적 연산(Atomic Operation)을 사용하여 이 문제를 해결하고, 병렬 알고리즘의 정확성을 보장하는 방법을 익힙니다.
### 준비: 데이터 생성

이미지의 밝기 값 분포를 나타내는 히스토그램을 계산하는 상황을 가정합니다. 히스토그램은 0부터 255까지 256개의 빈(bin)을 가집니다.
# 100만 픽셀을 가진 가상 이미지 데이터 (밝기 값만 1D 배열로)
num_pixels = 1_000_000
image_pixels = np.random.randint(0, 256, size=num_pixels, dtype=np.int32)

# GPU로 데이터 전송
d_image_pixels = cuda.to_device(image_pixels)
### 파트 1: 경쟁 상태 재현하기 (잘못된 커널)

먼저, 원자적 연산을 사용하지 않고 히스토그램을 계산하는 커널을 작성해봅시다.
@cuda.jit
def histogram_race_condition_kernel(pixels, hist_out):
    # 그리드-스트라이드 루프를 사용하여 모든 픽셀을 처리합니다.
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, pixels.shape[0], stride):
        pixel_value = pixels[i]
        # TODO: 경쟁 상태를 유발하는 코드를 작성하세요.
        # 단순히 해당 빈의 값을 1 증가시킵니다.
        hist_out[pixel_value] += 1

# 히스토그램 배열 초기화 및 실행
d_hist_1 = cuda.to_device(np.zeros(256, dtype=np.int32))
histogram_race_condition_kernel[256, 256](d_image_pixels, d_hist_1)

# 결과 확인
hist_1_result = d_hist_1.copy_to_host()
print(f"경쟁 상태 커널 결과 (총합): {hist_1_result.sum()}")
print(f"실제 픽셀 수: {num_pixels}")
print(f"결과가 정확한가? {hist_1_result.sum() == num_pixels}")
### 파트 2: 원자적 연산으로 문제 해결하기

이제 `cuda.atomic.add`를 사용하여 경쟁 상태를 해결해봅시다.
@cuda.jit
def histogram_atomic_kernel(pixels, hist_out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, pixels.shape[0], stride):
        pixel_value = pixels[i]
        # TODO: cuda.atomic.add를 사용하여 안전하게 빈의 값을 1 증가시키세요.
        # 사용법: cuda.atomic.add(배열, 인덱스, 증가값)
        cuda.atomic.add(hist_out, pixel_value, 1)

# 히스토그램 배열 초기화 및 실행
d_hist_2 = cuda.to_device(np.zeros(256, dtype=np.int32))
histogram_atomic_kernel[256, 256](d_image_pixels, d_hist_2)

# 결과 확인
hist_2_result = d_hist_2.copy_to_host()
print(f"원자적 연산 커널 결과 (총합): {hist_2_result.sum()}")
print(f"실제 픽셀 수: {num_pixels}")
print(f"결과가 정확한가? {hist_2_result.sum() == num_pixels}")
#### 분석 질문
1. 첫 번째 커널의 결과가 왜 부정확했나요? '읽기-수정-쓰기' 사이클과 연관지어 구체적인 시나리오를 설명해보세요.
2. `cuda.atomic.add`는 이 문제를 어떻게 해결했나요?
3. 자율주행 시스템에서 여러 센서 데이터를 종합하여 하나의 지도(Occupancy Grid Map)를 업데이트하는 상황을 상상해보세요. 이 상황에서 원자적 연산이 왜 중요할까요?


<img width="762" height="415" alt="image" src="https://github.com/user-attachments/assets/8c1967a1-414e-4eb5-a261-9d206f56a9b5" />

<img width="763" height="453" alt="image" src="https://github.com/user-attachments/assets/348916ed-f1a3-4e98-9aeb-a312aa0b26d5" />

<img width="762" height="590" alt="image" src="https://github.com/user-attachments/assets/3a606c62-34c1-4b85-8d49-8f97717fd63b" />

