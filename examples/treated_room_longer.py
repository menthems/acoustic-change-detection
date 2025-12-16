# corrected_treated_room.py
import numpy as np
from scipy.io import wavfile
import magnum as mn
import habitat_sim
import habitat_sim.agent
import habitat_sim.utils

# ---------- Configuration ----------
SCENE = "data/scene_datasets/sim/stages/treated_room.glb"
SCENE_DATASET = "data/scene_datasets/sim/simsetup.scene_dataset_config.json"
MATERIALS_JSON = "data/simsetup_material_config_soundcam.json"  # your materials file
OUTPUT_WAV = "data/output/transform_mic0.wav"
SAMPLE_RATE = 48000

# ---------- Create audio sensor spec ----------
audio_sensor_spec = habitat_sim.AudioSensorSpec()
audio_sensor_spec.uuid = "audio_sensor"
audio_sensor_spec.enableMaterials = True              # requires semantic mesh with semantic IDs
audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono
audio_sensor_spec.channelLayout.channelCount = 1
audio_sensor_spec.acousticsConfig.sampleRate = SAMPLE_RATE
audio_sensor_spec.acousticsConfig.indirect = True     # enable indirect paths (late reverberation)
# NOTE: audio_sensor_spec.position is the sensor offset relative to the agent.
# We set it to agent-local origin and place the agent at the mic world position.
audio_sensor_spec.position = mn.Vector3(0.0, 0.0, 0.0)
audio_sensor_spec.orientation = mn.Vector3(0.0, 0.0, 0.0)

# ---------- Simulator / Agent configuration ----------
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = SCENE
backend_cfg.scene_dataset_config_file = SCENE_DATASET
backend_cfg.load_semantic_mesh = True
backend_cfg.enable_physics = False

# Add the audio sensor to the agent BEFORE creating the Simulator
agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [audio_sensor_spec]

sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

# ---------- Create simulator ----------
sim = habitat_sim.Simulator(sim_cfg)

# ---------- Set agent (listener) world pose ----------
agent = sim.get_agent(0)
state = agent.get_state()

# Put the agent at the measured microphone world coordinates.
# Replace these with your measured microphone coordinates (x, y, z) in meters.
MIC_WORLD_POS = mn.Vector3(0.5445125, 1.277938, 3.152775)  # example from your data
state.position = MIC_WORLD_POS

# Set agent orientation (rotation) if needed (angle in radians around UP)
angle_radians = 2.0 * (np.pi / 2.0)   # example; change to your measured orientation
state.rotation = habitat_sim.utils.quat_from_angle_axis(angle_radians, habitat_sim.geo.UP)

agent.set_state(state)

# ---------- Get the audio sensor object (attached to the agent) ----------
# Use the agent's sensors to get the audio sensor instance that will be used for simulation.
audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]

# ---------- Load material mapping (must exist & match semantic mesh) ----------
# This should be done BEFORE triggering the audio simulator run (observations).
audio_sensor.setAudioMaterialsJSON(MATERIALS_JSON)

# ---------- Set the audio source (speaker) world position ----------
# Provide the source world coordinates (x, y, z)
SPEAKER_WORLD_POS = np.array([-0.3302, 1.1811, -0.17542], dtype=np.float32)
audio_sensor.setAudioSourceTransform(SPEAKER_WORLD_POS)

# ---------- (Optional) set a custom source waveform (recommended if you measured with a sweep) ----------
# If you have the same sweep used in your measurement, load and assign it here.
# If you don't, you can keep the default single-sample impulse the simulator uses.
# Example (uncomment and edit filename to use):
# sweep_sr, sweep = wavfile.read("my_measurement_sweep.wav")
# assert sweep_sr == SAMPLE_RATE, "sample rate mismatch"
# audio_sensor.setAudioSourceWaveform(sweep.astype(np.float32), sample_rate=sweep_sr)

# ---------- Request observations (this runs the audio simulator once and returns the RIR/mixture) ----------
obs_dict = sim.get_sensor_observations()
if "audio_sensor" not in obs_dict:
    raise RuntimeError("audio_sensor observation not returned; check sensor UUID and config.")
obs = np.array(obs_dict["audio_sensor"])  # shape (channels, samples) or (samples,) depending on API

# Ensure obs is 1-D mono time series for writing
if obs.ndim == 2:
    # Many builds return (channels, samples); for mono channels=1
    obs_mono = obs.squeeze()
else:
    obs_mono = obs

# Normalize if desired (avoid clipping)
maxv = np.max(np.abs(obs_mono))
if maxv > 0:
    # keep as float32 in [-1,1] for wavfile.write, or scale to int16 if you prefer.
    out_wave = (obs_mono / maxv * 0.95).astype(np.float32)
else:
    out_wave = obs_mono.astype(np.float32)

# ---------- Write WAV ----------
wavfile.write(OUTPUT_WAV, SAMPLE_RATE, out_wave)

print(f"Saved simulated audio to: {OUTPUT_WAV}")

# ---------- Cleanup ----------
sim.close()


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