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
