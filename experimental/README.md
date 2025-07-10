# Genesis Planner Demo

A simple planner implementation using the Genesis backend, designed to be modular and easy to understand.

## Design Principles

- **Avoid nested classes**: Keep classes simple and self-contained
- **Modular design**: Main classes are `Robot`, `Environment`, and `Planner`
- **Bottom-up approach**: Import robot → import scene → use planner
- **Mode switching**: Support both realistic and magic modes for grasping and movement

## Main Components

### Robot (`robot.py`)
- Simple robot class for Genesis backend
- Handles robot initialization and basic control operations
- Supports different action types: joint control, base velocity, magic movement

### Environment (`env.py`)
- Manages Genesis scene, robot, and provides environment interface
- Supports switching between realistic and magic modes
- Handles object addition and camera setup

### Planner (`planner.py`)
- Simple planner class with different strategies
- Supports random, heuristic, and LLM-based planning
- Easy to extend with new planning algorithms

## Features

### Grasping Modes
- **Realistic**: Uses physics-based grasping with contact detection
- **Magic**: Direct object manipulation without physics constraints

### Movement Modes
- **Realistic**: Physics-based movement with joint control
- **Magic**: Direct position/orientation setting

### Planner Types
- **Random**: Generates random actions
- **Heuristic**: Uses simple heuristics (e.g., move towards target)
- **LLM**: Placeholder for LLM-based planning (can be extended)

## Usage

### Basic Usage

```python
from genesis_planner import Robot, Environment, Planner, PlannerConfig, PlannerType

# Create robot configuration
robot_config = RobotConfig(urdf_path="path/to/robot.urdf")

# Create environment
env_config = EnvironmentConfig(robot_config=robot_config)
env = Environment(env_config)

# Create planner
planner_config = PlannerConfig(planner_type=PlannerType.HEURISTIC)
planner = Planner(planner_config, env)

# Run episode
observations = env.reset()
planner.reset()

for step in range(100):
    actions = planner.plan("Move to target", observations)
    step_result = env.step(actions)
    observations = step_result["observations"]
```

### Command Line Usage

```bash
# Basic usage
python experimental/genesis_planner/planner_demo.py

# With specific options
python experimental/genesis_planner/planner_demo.py \
    --planner_type heuristic \
    --grasp_mode realistic \
    --movement_mode magic \
    --instruction "Move to the target" \
    --max_steps 50 \
    --show_viewer

# Multiple episodes
python experimental/genesis_planner/planner_demo.py \
    --num_episodes 5 \
    --planner_type random
```

### Example Scripts

Run the example usage script to see different configurations:

```bash
python experimental/genesis_planner/example_usage.py
```

## Configuration Options

### Robot Configuration
- `urdf_path`: Path to robot URDF file
- `mass`: Robot mass
- `friction`: Friction coefficient
- `restitution`: Restitution coefficient
- `scale`: Robot scale
- `fixed`: Whether robot is fixed in place

### Environment Configuration
- `dt`: Simulation timestep
- `show_viewer`: Whether to show Genesis viewer
- `grasp_mode`: Realistic or magic grasping
- `movement_mode`: Realistic or magic movement
- `robot_config`: Robot configuration

### Planner Configuration
- `planner_type`: Random, heuristic, or LLM
- `max_steps`: Maximum planning steps
- `target_position`: Target position for heuristic planner

## Extending the Planner

### Adding New Planner Types

1. Add new type to `PlannerType` enum
2. Implement planning method in `Planner` class
3. Update `plan()` method to handle new type

```python
class PlannerType(Enum):
    RANDOM = "random"
    HEURISTIC = "heuristic"
    LLM = "llm"
    CUSTOM = "custom"  # New planner type

class Planner:
    def _custom_plan(self, instruction: str, observations: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # Implement custom planning logic
        pass
```

### Adding New Action Types

1. Add new action type to robot's `apply_action()` method
2. Update environment to handle new action types

### Adding New Object Types

1. Add new object type to environment's `add_object()` method
2. Define appropriate Genesis morphology and material

## Dependencies

- Genesis (for simulation backend)
- NumPy (for numerical operations)
- PyTorch (for device management)
- Dataclasses (for configuration)
- Enum (for mode definitions)

## Installation

1. Install Genesis following the [official documentation](https://genesis-world.readthedocs.io/)
2. Install Python dependencies:
   ```bash
   pip install numpy torch
   ```

## Notes

- This is an experimental implementation
- Some features require proper Genesis installation
- The LLM planner is a placeholder and needs actual LLM integration
- Error handling is simplified for demonstration purposes
- Performance may vary depending on Genesis installation and hardware

## Future Improvements

- Add actual LLM integration for LLM planner
- Implement more sophisticated grasping algorithms
- Add support for multi-agent scenarios
- Improve error handling and logging
- Add unit tests
- Add configuration file support
- Add visualization tools
