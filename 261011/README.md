# 폴더 구성
```
261011/
├── assignment02.py
├── README.md
└── src/
    └── images/
        └── sky.png
```

# 환경 준비
1. Python 3.14.0 설치 
2. VS Code 또는 PowerShell을 열고 폴더로 이동
3. 가상 환경 만들기

```powershell
py -3.14 -m venv .venv
```

4. 가상 환경 활성화

```powershell
.venv\Scripts\Activate.ps1
```

5. 패키지 설치

```powershell
pip install numpy opencv-python matplotlib scikit-image
```

# 실행 방법
1. 3번 과정에서 만든 가상 환경을 4번으로 활성화한 상태에서 실행합니다.
2. `src/images/sky.png` 경로를 상대 경로로 읽기 때문에, 반드시 `assignment02.py`가 있는 폴더 위치에서 아래 명령을 입력합니다.

```powershell
python assignment02.py
```

# 출력 확인
- 실습 결과가 순서대로 그림 창으로 출력됩니다. (총 10개)
- 현재 그림 창을 닫으면 다음 실습 결과 창이 열립니다.