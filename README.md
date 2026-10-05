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
| 0 | `main` | ant_reward(custom 보상 모듈)와 README 보관 | `e83a5d2` | `logs/rsl_rl/ant/2026-10-03_16-22-39_Test21` |
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
### Branch별 config

#### 0. `main` — ant_reward + README 보관

custom reward 함수와 발끝 kinematics 모듈이 처음 도입된 branch입니다. rough terrain, 마찰 랜덤화, 지면 센서, custom `RewardsCfg`가 모두 들어 있습니다.

| config | 파일 | 내용 |
| ------ | ---- | ---- |
| 학습 환경 | [ant_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py) `AntEnvCfg` | `num_envs=4096`, rough terrain(`boxes` 제외), 마찰 랜덤화, action=joint position |
| 평가 환경 | [ant_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py) `AntEnvCfg_PLAY` | `boxes` 지형 30×30, difficulty 0.8 고정, terrain curriculum off, foothold scan 시각화 |
| 센서 | [ant_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py) `MySceneCfg` | `height_scanner`(13×13), `foothold_scanner`(17×17), `contact_forces`(발 4개) |
| custom reward | [mdp/reward.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/mdp/reward.py) | `base_height_l2`, `feet_air_time`, `mechanical_power`, `feet_all_airborne`, `yaw_deviation_l2`, `swing_obstacle_clearance`, `stumble_without_lift` 등 |
| 발끝 kinematics | [mdp/ant_foot_kinematics.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/mdp/ant_foot_kinematics.py) | `FOOT_NAMES`, `AntFootKinematics`, `foot_tip_height` |
| PPO | [agents/rsl_rl_ppo_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/agents/rsl_rl_ppo_cfg.py) | `desired_kl=0.01`, `max_iterations=1000` |

#### 1. `legacy` — 도메인 random only

센서와 custom reward 없이, rough terrain + 마찰 랜덤화만 적용한 기준선입니다. reward는 baseline 7종을 그대로 사용합니다.

| config | 파일 | 내용 |
| ------ | ---- | ---- |
| 학습 환경 | [ant_env_cfg.py](https://github.com/roo-ham/IsaacLab_RS/blob/legacy/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py) `AntEnvCfg` | rough terrain, 센서 없음, baseline reward 7종, `clone_in_fabric=True` |
| 평가 환경 | [ant_env_play_cfg.py](https://github.com/roo-ham/IsaacLab_RS/blob/legacy/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_play_cfg.py) `AntEnvCfg_PLAY` | `boxes` 지형 30×30, difficulty 0.8 고정, curriculum off |
| domain randomization | [mdp/env_terms.py](https://github.com/roo-ham/IsaacLab_RS/blob/legacy/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/mdp/env_terms.py) | `randomize_discrete_friction`(마찰계수 0.6~1.2), `terrain_levels_speed`(지형 커리큘럼) |
| 관측 | `ant_env_cfg.py` `ObservationsCfg` | proprioception + `feet_body_forces` (지면 센서 없음) |
| PPO | [agents/rsl_rl_ppo_cfg.py](https://github.com/roo-ham/IsaacLab_RS/blob/legacy/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/agents/rsl_rl_ppo_cfg.py) | `desired_kl=0.01` |

#### 2. `legacy_sensor` — 도메인 random + 지면 센서

`legacy`에 지면 센서와 그에 따른 관측만 추가한 branch입니다. reward는 여전히 baseline 7종입니다.

| config | 파일 | 내용 |
| ------ | ---- | ---- |
| 학습 환경 | [ant_env_cfg.py](https://github.com/roo-ham/IsaacLab_RS/blob/legacy_sensor/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py) `AntEnvCfg` | rough terrain + 마찰 랜덤화 + 지면 센서, `clone_in_fabric=False` |
| 센서 | `ant_env_cfg.py` `MySceneCfg` | `height_scanner`(13×13=169), `foothold_scanner`(17×17=289), `contact_forces`(발 4개) |
| 관측 | `ant_env_cfg.py` `ObservationsCfg` | `base_height`를 지면 기준 torso 높이로 교체, `terrain_height_scan`(289) 추가 |
| mdp | [mdp/env_terms.py](https://github.com/roo-ham/IsaacLab_RS/blob/legacy_sensor/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/mdp/env_terms.py) | `torso_height_above_ground`, `torso_height_obs` 추가 |
| 평가 환경 | [ant_env_play_cfg.py](https://github.com/roo-ham/IsaacLab_RS/blob/legacy_sensor/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_play_cfg.py) `AntEnvCfg_PLAY` | `boxes` 지형 30×30, foothold scan 시각화 |
| PPO | [agents/rsl_rl_ppo_cfg.py](https://github.com/roo-ham/IsaacLab_RS/blob/legacy_sensor/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/agents/rsl_rl_ppo_cfg.py) | `desired_kl=0.01` |

#### 3. `group_ten` — 도메인 random + 센서 + custom reward

최신 학습이 수행된 branch입니다. 센서와 custom reward를 함께 쓰며, reward weight는 아래 III의 값입니다.

| config | 파일 | 내용 |
| ------ | ---- | ---- |
| 학습 환경 | [ant_env_cfg.py](https://github.com/roo-ham/IsaacLab_RS/blob/group_ten/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py) `AntEnvCfg` | rough terrain + 마찰 랜덤화 + 센서 + custom reward |
| 평가 환경 | 위 파일 `AntEnvCfg_PLAY` | `boxes` 지형, difficulty 0.8 고정, foothold scan 시각화 |
| custom reward | [mdp/reward.py](https://github.com/roo-ham/IsaacLab_RS/blob/group_ten/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/mdp/reward.py) | custom reward 함수 모음 (III 참고) |
| 발끝 kinematics | [mdp/ant_foot_kinematics.py](https://github.com/roo-ham/IsaacLab_RS/blob/group_ten/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/mdp/ant_foot_kinematics.py) | `foot_tip_height`, `foot_tip_height_local`, `foot_tip_state` |
| PPO | [agents/rsl_rl_ppo_cfg.py](https://github.com/roo-ham/IsaacLab_RS/blob/group_ten/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/agents/rsl_rl_ppo_cfg.py) | `desired_kl=0.015` |

#### 4. `group_ten_testreward` — group_ten + legacy reference reward

`group_ten`과 환경/보상 설정이 완전히 동일하고, 평가 시 legacy(cailab) reward 정의로 episode return을 함께 계산해 출력하는 기능만 추가되었습니다.

| config | 파일 | 내용 |
| ------ | ---- | ---- |
| 학습 환경 / reward | [ant_env_cfg.py](https://github.com/roo-ham/IsaacLab_RS/blob/group_ten_testreward/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py), [mdp/reward.py](https://github.com/roo-ham/IsaacLab_RS/blob/group_ten_testreward/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/mdp/reward.py) | `group_ten`과 동일 (weight 포함) |
| reference reward | [mdp/reference_reward.py](https://github.com/roo-ham/IsaacLab_RS/blob/group_ten_testreward/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/mdp/reference_reward.py) | cailab `upstream/main`(`e83a5d2`)의 Ant reward 정의를 동결해 별도 회계 (평가 전용) |
| play script | [play_one_episode.py](https://github.com/roo-ham/IsaacLab_RS/blob/group_ten_testreward/scripts/reinforcement_learning/rsl_rl/play_one_episode.py) | `=== LEGACY ... ===` / `=== OUR CODE ... ===` 섹션과 term별 통계 출력 |
| PPO | [agents/rsl_rl_ppo_cfg.py](https://github.com/roo-ham/IsaacLab_RS/blob/group_ten_testreward/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/agents/rsl_rl_ppo_cfg.py) | `desired_kl=0.015` |

---

## II. 실행 방법

### 공통 사항

- 학습(train)은 `Isaac-Ant-v0`, 평가/재생(play·test)은 `Isaac-Ant-Play-v0`에서 진행합니다.
  - train: `scripts/reinforcement_learning/rsl_rl/train.py`
  - play/test: `scripts/reinforcement_learning/rsl_rl/play_one_episode.py`
- 학습 log는 `logs/rsl_rl/ant/<run_name>` 아래에 쌓이며, 최종 checkpoint는 `model_999.pt`, 재생 영상은
  `logs/rsl_rl/ant/<run_name>/videos/play/rl-video-step-0.mp4` 입니다.
- 실행 전에 반드시 해당 branch로 이동합니다: `git checkout <branch>`

> ⚠️ **train 명령어는 최신 학습 branch인 `group_ten`에서 실행합니다.** (아래 3번 type의 학습이 여기에 해당)

### 1. 도메인 random only

| index | 제목 | branch | 학습 log 경로 | domain randomization | 바닥센서 여부 | Custom Reward 설계 여부 |
| ----- | ---- | ------ | ------------- | -------------------- | ------------- | ----------------------- |
| 1 | 도메인 random only | `legacy` | `logs/rsl_rl/ant/Terrain_And_Sensorless` | O (마찰계수 friction 0.6~1.2, 0.1 간격 7단계) | X | X |

#### 개요

`legacy` branch에서 실행 가능합니다. rough terrain 위에서 마찰계수만 도메인 랜덤화하고, 지면 센서와 custom reward는 쓰지 않습니다.
reward는 baseline 7종(`progress`, `alive`, `upright`, `move_to_target`, `action_l2`, `energy`, `joint_pos_limits`) 그대로이며,
정책은 proprioception과 발 wrench만으로 지형을 넘습니다. 참고로 같은 branch에 평지 학습 run(`Flat_And_Sensorless`)도 있습니다.

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

#### ant_report 파일

| 파일 | 목적 |
| ---- | ---- |
| [rl-video-step-0.mp4](<ant_report/1. 도메인랜덤 only/rl-video-step-0.mp4>) | `Terrain_And_Sensorless` 정책의 play 재생 영상 |
| [Screenshot from 2026-10-05 18-36-40.png](<ant_report/1. 도메인랜덤 only/Screenshot from 2026-10-05 18-36-40.png>) | TensorBoard 학습 곡선 (`mean_episode_length`, `mean_reward`, 999 iter) |
| [Screenshot from 2026-10-05 19-09-14.png](<ant_report/1. 도메인랜덤 only/Screenshot from 2026-10-05 19-09-14.png>) | `play_one_episode.py` 실행 결과 (episode reward total mean 18.101228, steps 852.24) |

### 2. 도메인 random + 센서

| index | 제목 | branch | 학습 log 경로 | domain randomization | 바닥센서 여부 | Custom Reward 설계 여부 |
| ----- | ---- | ------ | ------------- | -------------------- | ------------- | ----------------------- |
| 2 | 도메인 random + 센서 | `legacy_sensor` | `logs/rsl_rl/ant/Terrain_With_Sensor` | O (마찰계수 friction 0.6~1.2, 0.1 간격 7단계) | O (foothold 17×17 = 289개 + height 13×13 = 169개) | X |

#### 개요

`legacy_sensor` branch에서 실행 가능합니다. 1번 type에 지면 센서를 추가한 단계로,
torso 아래 지면 높이를 보는 `height_scanner`(169개)와 발이 디딜 넓은 영역을 보는 `foothold_scanner`(289개),
그리고 발 접촉/체공 시간을 재는 `contact_forces`(발 4개)를 사용합니다.
정책 관측에는 `terrain_height_scan`(289개)이 들어가고 `base_height`는 world z 대신 지면 기준 torso 높이로 바뀝니다. reward는 여전히 baseline 7종입니다.

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

#### ant_report 파일

| 파일 | 목적 |
| ---- | ---- |
| [rl-video-step-0.mp4](<ant_report/2. 도메인랜덤 + 센서/rl-video-step-0.mp4>) | `Terrain_With_Sensor` 정책의 play 재생 영상 |
| [Screenshot from 2026-10-05 19-50-25.png](<ant_report/2. 도메인랜덤 + 센서/Screenshot from 2026-10-05 19-50-25.png>) | TensorBoard 비교 곡선 (센서 없음 `Terrain_And_Sensorless` vs 센서 있음 `Terrain_With_Sensor`) |
| [Screenshot from 2026-10-05 19-50-08.png](<ant_report/2. 도메인랜덤 + 센서/Screenshot from 2026-10-05 19-50-08.png>) | `play_one_episode.py` 실행 결과 (episode reward total mean 20.108204, steps 833.36) |

### 3. 도메인 random + 센서 + custom reward

| index | 제목 | branch | 학습 log 경로 | domain randomization | 바닥센서 여부 | Custom Reward 설계 여부 |
| ----- | ---- | ------ | ------------- | -------------------- | ------------- | ----------------------- |
| 3 | 도메인 random + 센서 + custom reward | `group_ten_testreward` (train: **`group_ten`**) | `logs/rsl_rl/ant/2026-10-04_21-27-44_Test10` | O (마찰계수 friction 0.6~1.2, 0.1 간격 7단계) | O (foothold 17×17 = 289개 + height 13×13 = 169개) | O |

#### 개요

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

#### ant_report 파일

| 파일 | 목적 |
| ---- | ---- |
| [rl-video-step-0.mp4](<ant_report/3. 도메인랜덤 + 센서 + custom reward/rl-video-step-0.mp4>) | `Test10` 정책의 play 재생 영상 |
| [Screenshot from 2026-10-05 18-53-29.png](<ant_report/3. 도메인랜덤 + 센서 + custom reward/Screenshot from 2026-10-05 18-53-29.png>) | TensorBoard 학습 곡선 (`Test10`, 999 iter, `mean_reward` 224.9334) |
| [Screenshot from 2026-10-05 19-02-41.png](<ant_report/3. 도메인랜덤 + 센서 + custom reward/Screenshot from 2026-10-05 19-02-41.png>) | `play_one_episode.py`의 `=== LEGACY ... ===` 섹션 결과 (reference reward total mean -1.761047) |
| [별첨 - legacy reward 감점사유 (energy 소비 큼).png](<ant_report/3. 도메인랜덤 + 센서 + custom reward/별첨 - legacy reward 감점사유 (energy 소비 큼).png>) | legacy reward term별 통계 — `energy`의 `step_mean`이 50.11로 감점이 가장 큰 원인임을 보여주는 별첨 |

---

## III. custom reward 함수 (기존과 달라진 부분만 작성)

기준: **`group_ten`(최신 학습, type 3)의 custom reward**를, **기존 legacy 정의(`upstream/main` = `e83a5d2`, type 1/2의 reward)** 와 비교해
달라진 부분만 정리했습니다. `group_ten_testreward`의 reward도 아래와 동일합니다.

### 1. weight만 달라진 항목

| term | 함수 | 기존 weight | 변경 weight |
| ---- | ---- | ----------- | ----------- |
| `progress` | `humanoid.mdp.progress_reward` | 1.0 | 3.0 |
| `alive` | `mdp.is_alive` | 0.5 | 0.25 |
| `move_to_target` | `humanoid.mdp.move_to_target_bonus` | 0.5 | 0.01 |

### 2. 삭제된 항목 (있다가 삭제됨)

| term | 삭제 전 상태 | 경과 |
| ---- | ------------ | ---- |
| `upright` | `upright_posture_bonus`, weight 0.1 | 기존에 있다가 `group_ten`에서 삭제(주석 처리) |
| `action_l2` | `mdp.action_l2`, weight -0.005 | 기존에 있다가 삭제. action이 joint position target으로 바뀌어 default pose로 끌어당기는 문제 때문 |
| `joint_torques_l2` | `mdp.joint_torques_l2`, weight -1.0e-4 | `action_l2` 대체안으로 작성되었으나 등록 전에 삭제(주석 처리) |
| `feet_stumble` | `feet_stumble` 함수, weight -0.5 | 작성되었다가 아래 `stumble_without_lift`로 대체되며 삭제 |
| `foot_clearance` | `foot_clearance_reward` 함수, weight 1.0 | 작성되었으나 reward term 등록 전에 삭제(주석 처리). 그 자리를 `swing_obstacle_clearance`가 대체 |

### 3. 새로 만들어진 custom reward 함수

수식 없이, 함수의 코드 상 제목과 의미적 제목, weight만 적었습니다.

| 코드 상 제목 (함수명) | 의미적 제목 | weight |
| --------------------- | ----------- | ------ |
| `base_height_l2` | 지면 기준 torso 높이를 목표 높이에 맞추는 페널티 | -10.0 |
| `feet_air_time` | 발끝 높이와 전진 속도를 반영한 보폭(스텝) 보상 | 2.0 |
| `mechanical_power` | 실제 인가 토크 기준 소비 전력 페널티 (기존 `power_consumption` 대체) | -0.0067 |
| `feet_all_airborne` | 네 발이 모두 공중에 뜬 비행 상태 페널티 | -20.0 |
| `yaw_deviation_l2` | torso yaw가 진행 방향(+x)에서 벗어나는 것에 대한 페널티 | -2.0 |
| `swing_obstacle_clearance` | 스윙하는 발끝이 전방 지형(장애물)을 넘지 못할 때의 페널티 | -10.0 |
| `stumble_without_lift` | 벽을 미는 발이 들리지 않을 때의 페널티 | -5.0 |

보조 함수 (reward term에 직접 등록되지는 않음)

| 코드 상 제목 (함수명) | 의미적 제목 |
| --------------------- | ----------- |
| `foot_tip_state` | 발끝의 월드 좌표/속도 계산 (`swing_obstacle_clearance`, `stumble_without_lift`에서 사용) |
| `foot_tip_height`, `foot_tip_height_local` | 발끝 높이 및 국소 지면 기준 발끝 높이 계산 (`feet_air_time` 등에서 사용) |
| `AntFootKinematics`, `FOOT_NAMES` | 발끝 offset과 발 이름 정의 |

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
