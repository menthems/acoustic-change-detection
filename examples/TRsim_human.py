import quaternion

import habitat_sim
import numpy as np
from scipy.io import wavfile
import magnum as mn

import json
from pathlib import Path
import time

def modify_config_jsons(n, stage_template, tmp_stage, scene_dataset_template, tmp_scene_dataset):
    with open(stage_template, "r") as f:
        stage_cfg = json.load(f)
    stage_cfg["semantic_asset"] = f"../../stages/TRsim_{n}_semantic.ply"
    with open(tmp_stage, "w") as f:
        json.dump(stage_cfg, f, indent=2)
    
    with open(scene_dataset_template, "r") as f:
        scene_dataset_cfg = json.load(f)
    scene_dataset_cfg["semantic_scene_descriptor_instances"]["semantics_mapping"] = f"stages/TRsim_{n}.house"
    with open(tmp_scene_dataset, "w") as f:
        json.dump(scene_dataset_cfg, f, indent=2)

def pick_mic_location(mic_idx):
    mic_locations = {
        "0": [3.048000, 3.657600, -1.277938],
        "1": [1.930400, 3.1543625, -1.277938],
        "2": [1.285875, 3.111500, -1.277938],
        "3": [1.193800, 3.111500, -1.277938],
        "4": [0.5445125, 3.152775, -1.277938],
        "5": [-0.609600, 3.657600, -1.277938],
        "6": [-0.692150, 0.577850, -1.277938],
        "7": [1.939925, -0.4651375, -1.277938],
        "8": [2.447925, -0.4651375, -1.277938],
        "9": [3.048000, -0.4651375, -1.277938]
    } # ranges x, -y, -z

    mic_loc = mic_locations[str(mic_idx)]
    return mic_loc

def run_soundspaces_per_n(n, tmp_scene_dataset, indirect_ray_count=5000, mic_location=None, audio_samples=-1, save_wav=False):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "TRsim" #"data/scene_datasets/sim/stages/.glb" #"data/scene_datasets/sim/configs/scenes/not_empty.scene_instance.json" #"data/scene_datasets/sim/stages/cube_stage.glb" #"data/scene_datasets/trial/try/cube_15153.glb" #try5.glb"
    backend_cfg.scene_dataset_config_file = tmp_scene_dataset
    backend_cfg.load_semantic_mesh = True
    backend_cfg.enable_physics = False

    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.enableMaterials = True # make sure _semantic.ply file is in the scene folder
    audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono
    audio_sensor_spec.channelLayout.channelCount = 1
    audio_sensor_spec.position = mn.Vector3(0.0, 0.0, 0.0) #[0.692150, 1.277938, -0.577850]
    audio_sensor_spec.orientation = mn.Vector3(0.0, 0.0, 0.0)
    audio_sensor_spec.acousticsConfig.sampleRate = 48000
    audio_sensor_spec.acousticsConfig.indirect = True
    audio_sensor_spec.acousticsConfig.indirectRayCount = indirect_ray_count #50000000 # default: 5000; removes the vertical bands (due to directivity, not all rays come back to the microphone)
    audio_sensor_spec.acousticsConfig.indirectRayDepth = 500 #120 #adjusted_mp3d:200 #300
    audio_sensor_spec.acousticsConfig.sourceRayCount = 50000 # adjusted_mp3d:2000 #default: 200
    audio_sensor_spec.acousticsConfig.indirectSHOrder = 2
    audio_sensor_spec.acousticsConfig.globalVolume = 1.0 #3.0 #adjusted_mp3d:3.0#270.0 # default: 0.25; adjusts the amplitude of the waveform
    audio_sensor_spec.acousticsConfig.frequencyBands = 32 #adjusted_mp3d:12 # default: 4

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    sim.add_sensor(audio_sensor_spec)

    agent = sim.get_agent(0)
    state = agent.get_state()
    # angle = 1.5 * (np.pi / 2.0) # per 90 degrees
    # axis = np.array([0, 1, 0])
    # quat = habitat_sim.utils.quat_from_angle_axis(angle, axis)
    # state.rotation = quat
    state.position = mic_location
    agent.set_state(state)

    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(np.array([-0.3302, -0.17542, -1.1811]))#[-0.3302, 1.1811, -0.17542]))
    # audio_sensor.setAudioMaterialsJSON("data/simsetup_material_config_soundcam_othermaterial.json")
    audio_sensor.setAudioMaterialsJSON("data/TRsim_human_material_config.json")

    t_acoustic_start = time.perf_counter()
    obs = np.array(sim.get_sensor_observations()["audio_sensor"])
    t_acoustic_end = time.perf_counter()
    duration = t_acoustic_end - t_acoustic_start

    if save_wav:
        wavfile.write(f'data/output/TRsim/acoustic_parameters/deconvolved_{N}_{INDIRECT_RAY_COUNT}_mic{MIC_IDX}.wav', 48000, obs.T)

    sim.close()

    obs = obs[:, :audio_samples]
    
    return obs, duration

N = 1000                # amount of RIRs (up to 1000)            
MIC_CHANNELS = 10       # amount of mic channels
# MIC_IDX = 9             # for single mic RIR files
AUDIO_SAMPLES = 48000   # amount of samples to save in the .npy
SAVE_WAV = True        # whether or not to save separate .wav files per RIR
INDIRECT_RAY_COUNT = 50000
DATA = 'TRsim_human'
FOLDER = 'human' #'adjusted_mp3d'

stage_template = f"data/scene_datasets/{DATA}/configs/stages/TRsim_TEMPLATE.stage_config.json"
scene_dataset_template = f"data/scene_datasets/{DATA}/TRsim_TEMPLATE.scene_dataset_config.json"
tmp_stage = f"data/scene_datasets/{DATA}/configs/stages/TRsim.stage_config.json"
tmp_scene_dataset = f"data/scene_datasets/{DATA}/TRsim.scene_dataset_config.json"

deconvolved = np.zeros((N, 1, AUDIO_SAMPLES), dtype=np.float32)

for MIC_IDX in range(MIC_CHANNELS):
    mic_location = pick_mic_location(MIC_IDX)

    print(f"Creating {N} RIRs at an indirect ray count of {INDIRECT_RAY_COUNT}")
    print(f"For microphone {MIC_IDX} at location {mic_location}")

    for n in range(N):
        modify_config_jsons(n, stage_template, tmp_stage, scene_dataset_template, tmp_scene_dataset)
        rir, duration = run_soundspaces_per_n(n, tmp_scene_dataset, indirect_ray_count=INDIRECT_RAY_COUNT, mic_location=mic_location, audio_samples=AUDIO_SAMPLES, save_wav=SAVE_WAV)
        deconvolved[n] = rir
        print(f"[Timing] Acoustic simulation took " f"{duration:.3f} seconds")
        print(f"[Timing] Which means it would take {duration*1000/60:.1f} minutes for 1000 RIRs.")
        print(f"Finished RIR no.{n+1}")

    np.save(f"data/output/{DATA}/{FOLDER}/deconvolved_{N}_{INDIRECT_RAY_COUNT}_mic{MIC_IDX}.npy", deconvolved)
    print(f"Saved data/output/{DATA}/{FOLDER}/deconvolved_{N}_{INDIRECT_RAY_COUNT}_mic{MIC_IDX}.npy with shape:", deconvolved.shape)