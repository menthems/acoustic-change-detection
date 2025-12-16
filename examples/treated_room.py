import quaternion

import habitat_sim.sim
import numpy as np
from scipy.io import wavfile
import magnum as mn

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "data/scene_datasets/sim/stages/treated_room.glb" #"data/scene_datasets/sim/configs/scenes/not_empty.scene_instance.json" #"data/scene_datasets/sim/stages/cube_stage.glb" #"data/scene_datasets/trial/try/cube_15153.glb" #try5.glb"
backend_cfg.scene_dataset_config_file = "data/scene_datasets/sim/simsetup.scene_dataset_config.json"
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
audio_sensor_spec.acousticsConfig.indirectRayCount = 50000000 # default: 5000; removes the vertical bands (due to directivity, not all rays come back to the microphone)
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
wavfile.write('data/output/transform_mic0.wav', 48000, obs.T)

# SPEAKER
# [-0.3302, 0.17542, 1.1811]            -xzy: [0.3302, 1.1811, 0.17542]             yzx: [0.17542, 1.1811, -0.3302]         xz-y: [-0.3302, 1.1811, -0.17542]

# MICS
# 0: [3.048000, -3.657600, 1.277938]    -xzy: [-3.048000, 1.277938, -3.657600]      yzx: [-3.657600, 1.277938, 3.048000]    xz-y: [3.048000, 1.277938, 3.657600]
# 1: [1.930400, -3.1543625, 1.277938]   -xzy: [-1.930400, 1.277938, -3.1543625]     yzx: [-3.1543625, 1.277938, 1.930400]   xz-y: [1.930400, 1.277938, 3.1543625]
# 2: [1.285875, -3.111500, 1.277938]    -xzy: [-1.285875, 1.277938, -3.111500]      yzx: [-3.111500, 1.277938, 1.285875]    xz-y: [1.285875, 1.277938, 3.111500]
# 3: [1.193800, -3.111500, 1.277938]    -xzy: [-1.193800, 1.277938, -3.111500]      yzx: [-3.111500, 1.277938, 1.193800]    xz-y: [1.193800, 1.277938, 3.111500]
# 4: [0.5445125, -3.152775, 1.277938]   -xzy: [-0.5445125, 1.277938, -3.152775]     yzx: [-3.152775, 1.277938, 0.5445125]   xz-y: [0.5445125, 1.277938, 3.152775]
# 5: [-0.609600, -3.657600, 1.277938]   -xzy: [0.609600, 1.277938, -3.657600]       yzx: [-3.657600, 1.277938, -0.609600]   xz-y: [-0.609600, 1.277938, 3.657600]
# 6: [-0.692150, -0.577850, 1.277938]   -xzy: [0.692150, 1.277938, -0.577850]       yzx: [-0.577850, 1.277938, -0.692150]   xz-y: [-0.692150, 1.277938, 0.577850]
# 7: [1.939925, 0.4651375, 1.277938]    -xzy: [-1.939925, 1.277938, 0.4651375]      yzx: [0.4651375, 1.277938, 1.939925]    xz-y: [1.939925, 1.277938, -0.4651375]
# 8: [2.447925, 0.4651375, 1.277938]    -xzy: [-2.447925, 1.277938, 0.4651375]      yzx: [0.4651375, 1.277938, 2.447925]    xz-y: [2.447925, 1.277938, -0.4651375]
# 9: [3.048000, 0.4651375, 1.277938]    -xzy: [-3.048000, 1.277938, 0.4651375]      yzx: [1.277938, 3.048000, 0.4651375]    xz-y: [3.048000, 1.277938, -0.4651375]