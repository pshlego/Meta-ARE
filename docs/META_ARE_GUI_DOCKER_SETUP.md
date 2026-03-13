# Meta-ARE GUI: Docker 기반 세팅 가이드

## 목차

1. [개요](#1-개요)
2. [사전 준비](#2-사전-준비)
3. [Docker 이미지 빌드](#3-docker-이미지-빌드)
4. [컨테이너 실행](#4-컨테이너-실행)
5. [GUI 접속 및 사용](#5-gui-접속-및-사용)
6. [환경 변수 설정](#6-환경-변수-설정)
7. [주요 CLI 옵션](#7-주요-cli-옵션)
8. [트러블슈팅](#8-트러블슈팅)

---

## 1. 개요

Meta Agents Research Environments (ARE)는 Meta에서 개발한 AI 에이전트 벤치마크 플랫폼입니다.
Docker를 사용하면 Node.js, Python 등의 의존성을 직접 설치할 필요 없이 GUI를 바로 실행할 수 있습니다.

Docker 이미지는 multi-stage 빌드로 구성됩니다:

| Stage | 역할 |
|-------|------|
| Stage 1 (Node.js 23) | React 프론트엔드 빌드 |
| Stage 2 (Python 3.12) | Python 의존성 설치 |
| Stage 3 (Python 3.12) | Stage 1 + Stage 2 결합, 최종 런타임 이미지 |

---

## 2. 사전 준비

- **Docker** 설치 필요 (`docker --version`으로 확인)
- **디스크 공간** 최소 4GB 이상
- **소스 코드** 클론

```bash
# 이미 클론된 경우 생략
git clone https://github.com/facebookresearch/meta-agents-research-environments.git
cd meta-agents-research-environments
```

---

## 3. Docker 이미지 빌드

프로젝트 루트 디렉토리에서 실행합니다.

```bash
cd /Data2/v-parksungho/Meta-ARE

# 기본 빌드
docker build -t meta-are:latest .

# 캐시 없이 클린 빌드 (문제 발생 시)
docker build --no-cache -t meta-are:latest .

# 빌드 로그를 상세히 보고 싶을 때
docker build --progress=plain -t meta-are:latest .
```

> **참고**: 첫 빌드 시 Node.js, Python 의존성을 모두 다운로드하므로 시간이 다소 소요됩니다.

---

## 4. 컨테이너 실행

### 4.1 기본 실행

```bash
docker run -p 8080:8080 meta-are:latest
```

브라우저에서 `http://localhost:8080` 으로 접속합니다.

### 4.2 포트 충돌 시 (다른 포트 사용)

호스트의 8080 포트가 이미 사용 중인 경우 호스트 포트를 변경합니다.

```bash
# 호스트 8888 포트 → 컨테이너 8080 포트로 매핑
docker run -p 8888:8080 meta-are:latest
```

이 경우 `http://localhost:8888`로 접속합니다.

### 4.3 환경 변수 파일과 함께 실행

```bash
cp example.env .env
# .env 파일을 필요에 따라 수정한 후:
docker run -p 8080:8080 --env-file .env meta-are:latest
```

### 4.4 특정 시나리오를 미리 로드하여 실행

```bash
# 내장 시나리오 지정
docker run -p 8080:8080 meta-are:latest are-gui -s scenario_find_image_file

# HuggingFace 시나리오 지정
docker run -p 8080:8080 meta-are:latest are-gui -s "hf://datasets/meta-agents-research-environments/gaia2/adaptability/validation/scenario_universe_21_5e0gvz"
```

### 4.5 백그라운드 실행

```bash
docker run -d -p 8080:8080 --name meta-are meta-are:latest

# 로그 확인
docker logs -f meta-are

# 컨테이너 중지
docker stop meta-are

# 컨테이너 삭제
docker rm meta-are
```

---

## 5. GUI 접속 및 사용

컨테이너가 실행되면 브라우저에서 접속합니다:

- 기본: `http://localhost:8080`
- 포트 변경 시: `http://localhost:<지정한 포트>`

### 5.1 GUI 뷰 모드

GUI에는 두 가지 주요 워크스페이스가 있습니다:

#### Scenarios 워크스페이스

시나리오를 로드하고, DAG(방향 비순환 그래프)로 시각화하며 실행/관찰합니다.

- **시나리오 로드 방법**:
  - HuggingFace의 Gaia2 데이터셋에서 로드
  - 코드베이스 내장 시나리오에서 로드
  - JSON 파일 업로드 또는 URL로 로드
- **DAG 색상 코드**:
  - 파란색: 사용자 액션
  - 분홍색: 에이전트 액션
  - 초록색: 환경 이벤트 (예: 수신 이메일)
- **실행 제어**: Play / Pause / Stop 버튼으로 시뮬레이션 제어
- **내보내기**: DAG를 PNG로, 시나리오 트레이스를 JSON으로 저장 가능

#### Playground 워크스페이스

에이전트와 실시간 대화형 채팅 인터페이스입니다.

- 사전 정의된 프롬프트 또는 직접 입력으로 에이전트와 상호작용
- 에이전트가 시뮬레이션된 앱(이메일, 캘린더, 파일시스템 등)을 활용하는 과정 관찰
- 대화 내역을 JSON, Markdown, PDF로 내보내기 가능

#### Agent Logs 뷰

에이전트의 추론 과정을 단계별로 확인할 수 있습니다.

- 각 스텝의 액션, 도구 호출 상세 정보 확인 가능

### 5.2 애플리케이션 패널

좌측 네비게이션에서 시뮬레이션된 앱들의 데이터, 도구, 상태를 확인할 수 있습니다:

- 앱 데이터 뷰: UI 형식으로 앱 데이터 확인
- Tools 뷰: 해당 앱에서 사용 가능한 도구 목록
- App State 뷰: 앱 상태의 원시 JSON 데이터

---

## 6. 환경 변수 설정

`example.env`를 참고하여 `.env` 파일을 구성합니다.

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ARE_SIMULATION_SERVER_HOSTNAME` | (호스트명) | 서버 호스트명 |
| `ARE_SIMULATION_SERVER_PORT` | `8080` | 서버 포트 |
| `ARE_SIMULATION_SSL_CERT_PATH` | (없음) | SSL 인증서 경로 |
| `ARE_SIMULATION_SSL_KEY_PATH` | (없음) | SSL 키 경로 |
| `ARE_SCENARIO_DIRECTORIES` | (없음) | 커스텀 시나리오 디렉토리 (쉼표 구분) |
| `ALL_MODELS` | (없음) | 사용 가능 모델 목록 |

---

## 7. 주요 CLI 옵션

`are-gui` 명령에 전달할 수 있는 옵션들입니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-s / --scenario_id` | None | 실행할 시나리오 ID 또는 HuggingFace URL |
| `-h / --hostname` | `localhost` | 서버 호스트명 |
| `-p / --port` | `8080` | 서버 포트 |
| `-c / --certfile` | (없음) | SSL 인증서 경로 |
| `-k / --keyfile` | (없음) | SSL 키 경로 |
| `-d / --debug` | `False` | 디버그 모드 활성화 |
| `--ui_view` | None | 기본 뷰 모드 (`SCENARIOS` 또는 `PLAYGROUND`) |
| `--inactivity-limit` | `3600` | 세션 비활성 타임아웃 (초) |
| `--cleanup-interval` | `300` | 세션 정리 주기 (초) |
| `--dataset-path` | None | 로컬 JSON 시나리오 데이터셋 디렉토리 경로 |

Docker에서 CLI 옵션을 전달하려면:

```bash
docker run -p 8080:8080 meta-are:latest are-gui -s scenario_find_image_file --ui_view PLAYGROUND
```

---

## 8. 트러블슈팅

### 포트 충돌

```
Bind for 0.0.0.0:8080 failed: port is already allocated
```

**해결**: 호스트 포트를 변경합니다.

```bash
docker run -p 8888:8080 meta-are:latest
```

또는 기존에 8080을 사용하는 프로세스/컨테이너를 확인 후 중지합니다:

```bash
# 8080 포트를 사용 중인 프로세스 확인
sudo lsof -i :8080

# 기존 Docker 컨테이너가 사용 중인 경우
docker ps | grep 8080
docker stop <container_id>
```

### 빌드 실패 시

```bash
# 캐시 없이 클린 빌드
docker build --no-cache -t meta-are:latest .

# 빌드 로그 상세 출력
docker build --progress=plain -t meta-are:latest .
```

### 컨테이너가 바로 종료되는 경우

```bash
# 로그 확인
docker run -p 8080:8080 meta-are:latest  # foreground로 실행하여 로그 확인

# 또는 백그라운드 실행 후 로그 확인
docker logs <container_id>
```

### 기존 컨테이너/이미지 정리

```bash
# 중지된 컨테이너 삭제
docker container prune

# 사용하지 않는 이미지 삭제
docker image prune
```

---

## 9. HF 파일에 Agent Trace 추가 (world_logs 변환)

실험 결과의 HF 형식 JSON 파일에 `world_logs`가 비어있어 GUI에서 에이전트 트레이스가 표시되지 않는 경우, `convert_lite_to_world_logs.py` 스크립트를 사용하여 lite 형식의 대화 이력을 `world_logs`로 변환 후 주입할 수 있습니다.

### 스크립트 위치

```
/Data2/v-parksungho/meta_are_exp_results/convert_lite_to_world_logs.py
```

### 사용법

```bash
# 단일 시나리오 변환
python convert_lite_to_world_logs.py \
  --scenario-dir /path/to/scenario_universe_30_xxx

# 전체 배치 변환 (모든 scenario_universe_* 디렉토리)
python convert_lite_to_world_logs.py \
  --base-dir /path/to/merged_results

# 커스텀 출력 디렉토리 지정
python convert_lite_to_world_logs.py \
  --base-dir /path/to/merged_results \
  --output-dir /path/to/output
```

### 디렉토리 구조 요구사항

스크립트는 아래 구조를 전제합니다:

```
scenario_universe_30_xxx/
  raw/
    hf/   scenario_xxx_hash.json   (world_logs: [] 인 원본)
    lite/ scenario_xxx_hash.json   (per_agent_interaction_histories 포함)
```

hf와 lite 파일은 동일한 파일명(`{scenario_id}_{hash}.json`)을 공유해야 합니다.

### 출력

- 기본: 각 시나리오의 `raw/hf_with_logs/` 디렉토리에 변환된 파일 생성
- `--output-dir` 지정 시: 해당 경로 아래에 시나리오별 디렉토리 구조로 생성
- 원본 hf 파일은 수정되지 않음

### 변환 내용

lite 형식의 역할(role)이 다음 world_logs 엔트리로 변환됩니다:

| Lite role | world_logs log_type |
|-----------|-------------------|
| `system` | `system_prompt` |
| `user` | `task` |
| `assistant` | `step` + `llm_output` + `tool_call` |
| `tool-response` | `observation` |

마지막 tool call이 `AgentUserInterface__send_message_to_user`인 경우, `final_answer` + `end_task`가 추가로 생성됩니다.
