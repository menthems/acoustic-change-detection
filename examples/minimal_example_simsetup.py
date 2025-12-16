import quaternion
import habitat_sim.sim
import numpy as np
from scipy.io import wavfile

# --- Simulator configuration ---
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "data/scene_datasets/sim/configs/scenes/empty"
backend_cfg.scene_dataset_config_file = "data/scene_datasets/sim/simsetup.scene_dataset_config.json"
backend_cfg.load_semantic_mesh = True
backend_cfg.enable_physics = False

sem_cfg = habitat_sim.CameraSensorSpec()
sem_cfg.uuid = "semantic"
sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sem_cfg]

# Create simulator
sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(sim_cfg)

# --- Audio sensor ---
audio_sensor_spec = habitat_sim.AudioSensorSpec()
audio_sensor_spec.uuid = "audio_sensor"
audio_sensor_spec.enableMaterials = True
audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono
audio_sensor_spec.channelLayout.channelCount = 1
audio_sensor_spec.position = [0.0, 1.5, 0.0]
audio_sensor_spec.acousticsConfig.sampleRate = 16000
audio_sensor_spec.acousticsConfig.indirect = True
sim.add_sensor(audio_sensor_spec)

# --- Connect source & material config ---
audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
audio_sensor.setAudioSourceTransform(np.array([0.0, 0.0, 0.0]))
audio_sensor.setAudioMaterialsJSON("data/simsetup_material_config_RGB.json")

# --- Capture audio observation ---
obs = np.array(sim.get_sensor_observations()["audio_sensor"])
wavfile.write("data/simsetup_output.wav", 16000, obs.T)
