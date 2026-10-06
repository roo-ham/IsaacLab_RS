# 실습 과제 1: 처음 보는 환경에서도 잘 걷는 Ant 만들기
- 로보틱스시뮬레이션 10조
  - 팀원: 박현우, 이한얼, 정재욱

**Isaac Lab** (https://github.com/isaac-sim/IsaacLab & https://github.com/cailab-hy/IsaacLab_RS) 기반으로 Ant 보행 정책을 학습/평가한 저장소입니다.

## task 목록
- 학습(task): `Isaac-Ant-v0`
- 평가/재생(task): `Isaac-Ant-Play-v0`

## I. 각 Branch의 정보

### Branch 요약 표

| index | branch | 목적 | 분기 기준 | 대표 학습 log 경로 |
| ----- | ------ | ---- | --------- | ------------------ |
| 0 | `main` | ant_reward(custom 보상 모듈)와 README 보관 | `e83a5d2` |  |
| 1 | `legacy` | 도메인 random only (센서·custom reward 없음) | `e83a5d2` | `logs/rsl_rl/ant/Terrain_And_Sensorless` |
| 2 | `legacy_sensor` | 도메인 random + 지면 센서 | `e83a5d2` | `logs/rsl_rl/ant/Terrain_With_Sensor` |
| 3 | `group_ten` | 도메인 random + 센서 + custom reward (최신 학습) | `main` (`9616bac`) | `logs/rsl_rl/ant/2026-10-04_21-27-44_Test10` |
| 4 | `group_ten_testreward` | `group_ten` + legacy reference reward 회계 (type 3 실행 branch) | `main` (`9616bac`) | `logs/rsl_rl/ant/2026-10-04_21-27-44_Test10` |

### 개요

- 모든 branch는 `upstream/main`(`e83a5d2`)의 Ant task를
  출발점으로 하며, 실험 목적에 따라 다음 두 갈래로 파생되었습니다.
  - legacy reward (보상함수는 `cailab-hy/IsaacLab_RS`과 동일)
    - **`legacy`(domain randomization only)**
    - **`legacy_sensor`(domain randomization + 지면 센서)**
  - custom reward (도메인 랜덤화, 센서, custom reward)
    - **`group_ten`(학습용)**
    - **`group_ten_testreward`(시연용)**
- `main` branch의 목적은 **ant_reward(custom Ant 보상 모듈)와 이 README를 보관하는 것**입니다.
  실제 최신 학습은 `group_ten`에서 수행되며, `group_ten_testreward`는 `group_ten`에 legacy reward 회계(reference reward)만 추가한 평가용 branch입니다.
- 공통 config 요소
  - 지형: `ROUGH_TERRAINS_CFG` 기반 생성 지형. 학습에서는 `boxes`를 제외한 sub-terrain을 쓰고, 평가(`PLAY`)에서만 held-out인 `boxes`를 사용합니다.
  - action: `JointPositionAction` (scale 0.5, `use_default_offset=True`)
  - actuator: `IdealPDActuatorCfg`
    - `stiffness`(P 제어기 값) = 20
    - `effort_limit`(최대 토크) = 10
  - PPO Hyper Parameter
    - legacy·legacy_sensor는 `desired_kl=0.01`
    - group_ten·group_ten_testreward는 `desired_kl=0.015`

---

## II. 실행 방법

| index | 제목 | branch | 학습 log 경로 | domain randomization | 바닥센서 여부 | Custom Reward 설계 여부 |
| ----- | ---- | ------ | ------------- | -------------------- | ------------- | ----------------------- |
| 1 | 도메인 random only | `legacy` | `logs/rsl_rl/ant/Terrain_And_Sensorless` | O (마찰계수 friction 0.6~1.2, 0.1 간격 7단계) | X | X |
| 2 | 도메인 random + 센서 | `legacy_sensor` | `logs/rsl_rl/ant/Terrain_With_Sensor` | O (마찰계수 friction 0.6~1.2, 0.1 간격 7단계) | O (foothold 17×17 = 289개 + height 13×13 = 169개) | X |
| 3 | 도메인 random + 센서 + custom reward | `group_ten_testreward` (train: **`group_ten`**) | `logs/rsl_rl/ant/2026-10-04_21-27-44_Test10` | O (마찰계수 friction 0.6~1.2, 0.1 간격 7단계) | O (foothold 17×17 = 289개 + height 13×13 = 169개) | O |

### 1. 도메인 random only

```bash
# 0) branch 준비
git checkout legacy

# 1) train - Isaac-Ant-v0
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --task Isaac-Ant-v0 --seed 42 \
  --run_name=Terrain_And_Sensorless --max_iterations 1000

# 2) play / test - Isaac-Ant-Play-v0
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Play-v0 --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/Terrain_And_Sensorless/model_999.pt \
  --video --video_length 960
```

### 2. 도메인 random + 센서

```bash
# 0) branch 준비
git checkout legacy_sensor

# 1) train - Isaac-Ant-v0
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --task Isaac-Ant-v0 --seed 42 \
  --run_name=Terrain_With_Sensor --max_iterations 1000

# 2) play / test - Isaac-Ant-Play-v0
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Play-v0 --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/Terrain_With_Sensor/model_999.pt \
  --video --video_length 960
```

### 3. 도메인 random + 센서 + custom reward

`group_ten_testreward` branch에서 실행 가능합니다. 2번 type에 custom reward를 설계/튜닝한 단계이며,
환경과 reward 정의는 `group_ten`과 완전히 동일합니다. `group_ten_testreward`는 여기에 더해 평가 시
legacy(`e83a5d2`) reward 정의로 episode return을 따로 계산해 `=== LEGACY ... ===` / `=== OUR CODE ... ===`로 나누어 출력합니다.
**train은 최신 학습이 수행된 `group_ten` branch에서 실행하고, play/test는 `group_ten_testreward` branch에서 실행합니다.**

```bash
# 1) train - Isaac-Ant-v0
#    ⚠️ train은 반드시 group_ten branch에서 실행합니다. (최신 학습)
git checkout group_ten

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --task Isaac-Ant-v0 --seed 42 \
  --run_name=Test10 --max_iterations 1000

# 2) play / test - Isaac-Ant-Play-v0
#    평가(reference reward 회계)는 group_ten_testreward branch에서 실행합니다.
git checkout group_ten_testreward

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Play-v0 --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-10-04_21-27-44_Test10/model_999.pt \
  --video --video_length 960
```

---

## III. ant_report 폴더

### 1. 도메인 random only

| 파일 | 목적 |
| ---- | ---- |
| [rl-video-step-0.mp4](<ant_report/1. 도메인랜덤 only/rl-video-step-0.mp4>) | `Terrain_And_Sensorless` 정책의 play 재생 영상 |
| [Screenshot from 2026-10-05 18-36-40.png](<ant_report/1. 도메인랜덤 only/Screenshot from 2026-10-05 18-36-40.png>) | TensorBoard 학습 곡선 (`mean_episode_length`, `mean_reward`, 999 iter) |
| [Screenshot from 2026-10-05 19-09-14.png](<ant_report/1. 도메인랜덤 only/Screenshot from 2026-10-05 19-09-14.png>) | `play_one_episode.py` 실행 결과 (episode reward total mean 18.101228, steps 852.24) |

### 2. 도메인 random + 센서

| 파일 | 목적 |
| ---- | ---- |
| [rl-video-step-0.mp4](<ant_report/2. 도메인랜덤 + 센서/rl-video-step-0.mp4>) | `Terrain_With_Sensor` 정책의 play 재생 영상 |
| [Screenshot from 2026-10-05 19-50-25.png](<ant_report/2. 도메인랜덤 + 센서/Screenshot from 2026-10-05 19-50-25.png>) | TensorBoard 비교 곡선 (센서 없음 `Terrain_And_Sensorless` vs 센서 있음 `Terrain_With_Sensor`) |
| [Screenshot from 2026-10-05 19-50-08.png](<ant_report/2. 도메인랜덤 + 센서/Screenshot from 2026-10-05 19-50-08.png>) | `play_one_episode.py` 실행 결과 (episode reward total mean 20.108204, steps 833.36) |

### 3. 도메인 random + 센서 + custom reward

| 파일 | 목적 |
| ---- | ---- |
| [rl-video-step-0.mp4](<ant_report/3. 도메인랜덤 + 센서 + custom reward/rl-video-step-0.mp4>) | `Test10` 정책의 play 재생 영상 |
| [Screenshot from 2026-10-05 18-53-29.png](<ant_report/3. 도메인랜덤 + 센서 + custom reward/Screenshot from 2026-10-05 18-53-29.png>) | TensorBoard 학습 곡선 (`Test10`, 999 iter, `mean_reward` 224.9334) |
| [Screenshot from 2026-10-05 19-02-41.png](<ant_report/3. 도메인랜덤 + 센서 + custom reward/Screenshot from 2026-10-05 19-02-41.png>) | `play_one_episode.py`의 `=== LEGACY ... ===` 섹션 결과 (reference reward total mean -1.761047) |
| [별첨 - legacy reward 감점사유 (energy 소비 큼).png](<ant_report/3. 도메인랜덤 + 센서 + custom reward/별첨 - legacy reward 감점사유 (energy 소비 큼).png>) | legacy reward term별 통계 — `energy`의 `ep_return_mean`이 -24.83로 감점이 가장 큰 원인임을 보여주는 별첨 |

---

## IV. custom reward 함수

기준: **`group_ten`(최신 학습, type 3)의 custom reward**를, **기존 legacy 정의** 와 비교해 달라진 부분만 정리했습니다.

### 1. weight만 달라진 항목

| term | 함수 | 기존 weight | 변경 weight |
| ---- | ---- | ----------- | ----------- |
| `progress` | `humanoid.mdp.progress_reward` | 1.0 | 3.0 |
| `alive` | `mdp.is_alive` | 0.5 | 0.25 |
| `move_to_target` | `humanoid.mdp.move_to_target_bonus` | 0.5 | 0.01 |

### 2. 삭제된 항목

| term | 삭제 전 상태 | 경과 |
| ---- | ------------ | ---- |
| `upright` | `upright_posture_bonus`, weight 0.1 | 기존에 있다가 `group_ten`에서 삭제(주석 처리) |
| `action_l2` | `mdp.action_l2`, weight -0.005 | 기존에 있다가 삭제. action이 joint position target으로 바뀌어 default pose로 끌어당기는 문제 때문 |
| `joint_torques_l2` | `mdp.joint_torques_l2`, weight -1.0e-4 | `action_l2` 대체안으로 작성되었으나 등록 전에 삭제(주석 처리) |
| `feet_stumble` | `feet_stumble` 함수, weight -0.5 | 작성되었다가 아래 `stumble_without_lift`로 대체되며 삭제 |
| `foot_clearance` | `foot_clearance_reward` 함수, weight 1.0 | 작성되었으나 reward term 등록 전에 삭제(주석 처리). 그 자리를 `swing_obstacle_clearance`가 대체 |

### 3. 새로 만들어진 custom reward 함수

| 코드 상 제목 (함수명) | 의미적 제목 | weight |
| --------------------- | ----------- | ------ |
| `base_height_l2` | 지면 기준 torso 높이를 목표 높이에 맞추는 페널티 | -10.0 |
| `feet_air_time` | 발끝 높이와 전진 속도를 반영한 보폭(스텝) 보상 | 2.0 |
| `mechanical_power` | 실제 인가 토크 기준 소비 전력 페널티 (기존 `power_consumption` 대체) | -0.0067 |
| `feet_all_airborne` | 네 발이 모두 공중에 뜬 비행 상태 페널티 | -20.0 |
| `yaw_deviation_l2` | torso yaw가 진행 방향(+x)에서 벗어나는 것에 대한 페널티 | -2.0 |
| `swing_obstacle_clearance` | 스윙하는 발끝이 전방 지형(장애물)을 넘지 못할 때의 페널티 | -10.0 |
| `stumble_without_lift` | 벽을 미는 발이 들리지 않을 때의 페널티 | -5.0 |

### 4. 기존 core mdp 함수를 재사용해 새로 추가된 reward term

새 함수는 아니지만 기존 정의에 없던 항목입니다.

| term | 함수 | weight |
| ---- | ---- | ------ |
| `action_rate_l2` | `mdp.action_rate_l2` | -0.01 |
| `lin_vel_z_l2` | `mdp.lin_vel_z_l2` | -0.5 |
| `termination` | `mdp.is_terminated` | -100.0 |
| `ang_vel_xy_l2` | `mdp.ang_vel_xy_l2` | -0.05 |

> 참고: `energy`(`mechanical_power`)는 weight 자체도 기존 -0.05에서 -0.05/7.5로 바뀌었고, 사용 함수도
> `humanoid.mdp.power_consumption`에서 위 `mechanical_power`로 교체되었습니다.
