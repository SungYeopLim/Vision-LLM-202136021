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
pip install numpy
```

# 실행 방법
1. 3번 과정에서 만든 가상 환경을 4번으로 활성화한 상태에서 실행합니다.
2. 폴더 위치에서 아래 명령을 입력합니다.

```powershell
python assignment01.py
```