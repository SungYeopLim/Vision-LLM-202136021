# Assignment 02

## 파일 구조

- **assignment02.py**: 농구공 이미지에 LAB color segmentation과 CLAHE 강화를 적용한 뒤, Morphological operation으로 최종 binary mask를 만들고 가장 큰 외곽선을 선택하는 코드
- **README.md**: 과제 설명 문서

```
Vision-LLM-202136021/
├── src/
│   └── images/
│       └── basketball_crop.jpg
└── 261017/
    ├── assignment02.py
    └── README.md
```

<br>

## 실행 방식

1. Python 3.14.0을 설치하고, VS Code 또는 PowerShell을 열어 `261017` 폴더로 이동합니다.
2. 가상 환경을 만듭니다.

```powershell
py -3.14 -m venv .venv
```

3. 가상 환경을 활성화합니다.

```powershell
.venv\Scripts\Activate.ps1
```

4. 패키지를 설치합니다.

```powershell
pip install numpy opencv-python matplotlib
```

5. 이미지 경로가 `../src/images/...` 상대 경로이므로, 가상 환경을 활성화한 상태로 `261017` 폴더 위치에서 아래 명령을 입력합니다.

```powershell
python assignment02.py
```

<br>

## 기타 참고 사항

### 최종 binary mask 선택 이유

최종 binary mask는 **Opening 2회 → Closing 1회** (커널: 5x5 타원) 순서로 생성했습니다.

Opening과 Closing의 횟수와 순서를 바꿔 가며 여러 조합을 직접 출력해 봤는데, 노이즈가 얼마나 남는지와 공 테두리가 얼마나 깔끔한지를 눈으로 비교했을 때 이 조합이 가장 좋아 보였습니다.

1. **Opening을 먼저 한 이유**: CLAHE로 대비를 키운 mask를 보니 조명 반사 때문에 생긴 가는 줄무늬와 점 노이즈가 꽤 많았습니다. Closing을 먼저 해 보니 이 노이즈가 공에 달라붙어서 외곽선이 공보다 훨씬 크게 잡혔고, 그래서 노이즈부터 깎아내는 Opening을 먼저 하기로 했습니다.
2. **Opening을 2번 한 이유**: 1번만 하면 공 위쪽에 자잘한 조각이 아직 남아 있었고, 2번 하니까 거의 다 정리됐습니다. 3번까지 해 보니 노이즈는 더 줄지 않고 오히려 공이 조금씩 깎여서 2번에서 멈췄습니다.
3. **마지막에 Closing을 1번 한 이유**: Opening으로 깎이면서 공 테두리 안쪽이 살짝 패인 곳이 생겨서, Closing으로 메워 외곽선을 매끄럽게 다듬었습니다. 2번 이상 해도 눈에 띄는 차이가 없어서 1번만 했습니다.