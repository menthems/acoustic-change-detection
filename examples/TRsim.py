import quaternion

import habitat_sim
import numpy as np
from scipy.io import wavfile
import magnum as mn

import json
from pathlib import Path

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

def run_soundspaces_per_n(n, tmp_stage, tmp_scene_dataset):
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
    audio_sensor_spec.acousticsConfig.indirectRayCount = 5000 #50000000 # default: 5000; removes the vertical bands (due to directivity, not all rays come back to the microphone)
    audio_sensor_spec.acousticsConfig.indirectRayDepth = 300
    audio_sensor_spec.acousticsConfig.sourceRayCount = 2000 # default: 200
    audio_sensor_spec.acousticsConfig.indirectSHOrder = 1
    audio_sensor_spec.acousticsConfig.globalVolume = 270.0 # default: 0.25; adjusts the amplitude of the waveform
    audio_sensor_spec.acousticsConfig.frequencyBands = 12 # default: 4

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    sim.add_sensor(audio_sensor_spec)

    agent = sim.get_agent(0)
    state = agent.get_state()
    angle = 0.0 #1.5 * (np.pi / 2.0) # per 90 degrees
    axis = np.array([0, 1, 0])
    quat = habitat_sim.utils.quat_from_angle_axis(angle, axis)
    state.rotation = quat
    state.position = mn.Vector3(-0.692150, 1.277938, 0.577850)
    agent.set_state(state)

    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(np.array([-0.3302, 1.1811, -0.17542]))#[-0.3302, 1.1811, -0.17542]
    audio_sensor.setAudioMaterialsJSON("data/simsetup_material_config_soundcam_othermaterial.json")

    obs = np.array(sim.get_sensor_observations()["audio_sensor"])
    wavfile.write('data/output/TRsim_{n}.wav', 48000, obs.T)

    sim.close()

stage_template = "data/scene_datasets/TRsim/configs/stages/TRsim_TEMPLATE.stage_config.json"
scene_dataset_template = "data/scene_datasets/TRsim/TRsim_TEMPLATE.scene_dataset_config.json"
tmp_stage = "data/scene_datasets/TRsim/configs/stages/TRsim.stage_config.json"
tmp_scene_dataset = "data/scene_datasets/TRsim/TRsim.scene_dataset_config.json"

for n in range(1):
    modify_config_jsons(n, stage_template, tmp_stage, scene_dataset_template, tmp_scene_dataset)
    run_soundspaces_per_n(n, tmp_stage, tmp_scene_dataset)
    print(f"Finished {n}")