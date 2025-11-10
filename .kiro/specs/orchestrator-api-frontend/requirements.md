# Requirements Document

## Introduction

This feature transforms the existing realtime orchestrator system into a client-server architecture with a REST/WebSocket API backend and an interactive web-based frontend. The orchestrator currently runs as a standalone Python application that generates OSC messages for audio synthesis based on tension zones. The new system will expose the orchestrator's controls via API endpoints while maintaining beat-accurate timing, and provide an intuitive 2D canvas interface where users can control tension zones through spatial interaction.

## Glossary

- **Orchestrator**: The Python-based realtime audio control system that generates OSC messages based on tension values and musical parameters
- **Tension Zone**: A named region (zone1, zone2, zone3) that influences harmonic and melodic decisions, with values ranging from 0.0 to 1.0
- **OSC (Open Sound Control)**: A protocol for networking sound synthesizers and other multimedia devices
- **API Server**: The FastAPI-based HTTP/WebSocket server that wraps the orchestrator and exposes control endpoints
- **Frontend Client**: The web-based user interface with a 2D canvas for spatial tension control
- **Canvas Point**: A user-controlled draggable point on the 2D canvas representing the current position
- **Zone Anchor**: Fixed points on the canvas (left, top, right) representing the three tension zones
- **Distance-Based Tension**: Tension values calculated inversely proportional to the Euclidean distance between the canvas point and zone anchors
- **Beat-Accurate Timing**: The orchestrator's ability to maintain precise musical timing synchronized to BPM
- **WebSocket Connection**: A persistent bidirectional communication channel between frontend and backend for realtime updates

## Requirements

### Requirement 1

**User Story:** As a musician, I want the orchestrator to run as an independent API service, so that I can control it from multiple clients without coupling the audio engine to any specific interface.

#### Acceptance Criteria

1. WHEN THE API Server starts, THE API Server SHALL initialize the Orchestrator instance and begin the beat-accurate timing loop
2. WHILE THE Orchestrator runs, THE API Server SHALL maintain beat-accurate timing independent of API request processing
3. THE API Server SHALL expose HTTP endpoints for configuration operations (BPM, key, beats per bar)
4. THE API Server SHALL expose HTTP endpoints for control operations (tension updates, chord overrides, drum fills)
5. THE API Server SHALL run on a configurable port separate from OSC output ports

### Requirement 2

**User Story:** As a musician, I want to receive realtime feedback about the orchestrator state, so that I can see current tension values, BPM, and musical parameters as they change.

#### Acceptance Criteria

1. THE API Server SHALL expose a WebSocket endpoint for realtime state updates
2. WHEN a tension value changes, THE API Server SHALL broadcast the updated tension values to all connected WebSocket clients within 50 milliseconds
3. WHEN the BPM changes, THE API Server SHALL broadcast the updated BPM to all connected WebSocket clients within 50 milliseconds
4. WHEN a bar downbeat occurs, THE API Server SHALL broadcast the current bar index to all connected WebSocket clients within 100 milliseconds
5. THE API Server SHALL send current orchestrator state to WebSocket clients immediately upon connection

### Requirement 3

**User Story:** As a musician, I want a web-based interface with a 2D canvas, so that I can control tension zones through intuitive spatial interaction rather than abstract sliders.

#### Acceptance Criteria

1. THE Frontend Client SHALL render a 2D canvas with minimum dimensions of 600x600 pixels
2. THE Frontend Client SHALL display three Zone Anchors at fixed positions: left (10%, 50%), top (50%, 10%), and right (90%, 50%)
3. THE Frontend Client SHALL display a draggable Canvas Point that users can move with mouse or touch input
4. THE Frontend Client SHALL visually distinguish Zone Anchors from the Canvas Point using different colors and sizes
5. THE Frontend Client SHALL render connection lines between the Canvas Point and each Zone Anchor

### Requirement 4

**User Story:** As a musician, I want tension values to be calculated based on my canvas position, so that moving closer to a zone anchor increases that zone's tension naturally.

#### Acceptance Criteria

1. WHEN THE Canvas Point moves, THE Frontend Client SHALL calculate the Euclidean distance to each Zone Anchor
2. THE Frontend Client SHALL calculate tension values inversely proportional to distance using the formula: tension = max(0, 1 - (distance / max_distance))
3. WHERE a Zone Anchor is more than 70% of the canvas diagonal distance away, THE Frontend Client SHALL set that zone's tension to 0.0
4. THE Frontend Client SHALL normalize tension values so the maximum tension across all zones equals 1.0
5. WHEN tension values change, THE Frontend Client SHALL send updated values to the API Server via HTTP POST within 100 milliseconds

### Requirement 5

**User Story:** As a musician, I want to adjust global parameters like BPM and key from the frontend, so that I have complete control over the orchestrator without switching tools.

#### Acceptance Criteria

1. THE Frontend Client SHALL provide input controls for BPM (range: 40-240)
2. THE Frontend Client SHALL provide a dropdown selector for musical key (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
3. THE Frontend Client SHALL provide input controls for beats per bar (range: 1-12)
4. WHEN a user changes BPM, THE Frontend Client SHALL send the new value to the API Server via HTTP POST
5. WHEN a user changes key, THE Frontend Client SHALL send the new value to the API Server via HTTP POST

### Requirement 6

**User Story:** As a musician, I want to trigger drum fills and chord overrides from the frontend, so that I can add musical variations during performance.

#### Acceptance Criteria

1. THE Frontend Client SHALL provide buttons to trigger drum fills with presets (snare, toms, hats)
2. THE Frontend Client SHALL provide an input field and button to queue chord overrides (format: "C_M", "F_M7", etc.)
3. WHEN a user triggers a drum fill, THE Frontend Client SHALL send the fill parameters to the API Server via HTTP POST
4. WHEN a user queues a chord override, THE Frontend Client SHALL send the chord symbol to the API Server via HTTP POST
5. THE Frontend Client SHALL display visual feedback confirming that commands were sent successfully

### Requirement 7

**User Story:** As a developer, I want the API to handle concurrent requests safely, so that multiple clients or rapid user interactions don't corrupt the orchestrator state.

#### Acceptance Criteria

1. THE API Server SHALL use thread-safe mechanisms to update Orchestrator state from API requests
2. WHEN multiple tension update requests arrive concurrently, THE API Server SHALL process them sequentially without data races
3. THE API Server SHALL validate all incoming parameters and return HTTP 400 for invalid values
4. THE API Server SHALL return HTTP 500 with error details when internal errors occur
5. THE API Server SHALL log all API requests and errors for debugging purposes

### Requirement 8

**User Story:** As a musician, I want the frontend to be responsive and visually clear, so that I can use it effectively during live performance.

#### Acceptance Criteria

1. THE Frontend Client SHALL update the canvas display at minimum 30 frames per second during dragging
2. THE Frontend Client SHALL display current tension values numerically alongside the canvas
3. THE Frontend Client SHALL use distinct colors for each Zone Anchor (zone1: blue, zone2: green, zone3: red)
4. THE Frontend Client SHALL scale the canvas responsively to fit the browser window while maintaining aspect ratio
5. THE Frontend Client SHALL display connection status to the WebSocket (connected/disconnected)

### Requirement 9

**User Story:** As a system administrator, I want the API server to be configurable via environment variables or config files, so that I can deploy it in different environments without code changes.

#### Acceptance Criteria

1. THE API Server SHALL read configuration from environment variables or a config file
2. THE API Server SHALL support configuration of API port (default: 8080)
3. THE API Server SHALL support configuration of OSC output address and ports
4. THE API Server SHALL support configuration of CORS origins for the frontend
5. WHERE configuration is missing, THE API Server SHALL use sensible defaults and log warnings

### Requirement 10

**User Story:** As a musician, I want the system to gracefully handle disconnections, so that temporary network issues don't crash the audio engine.

#### Acceptance Criteria

1. WHEN THE WebSocket connection drops, THE Frontend Client SHALL attempt to reconnect automatically every 3 seconds
2. WHEN THE Frontend Client reconnects, THE API Server SHALL send the current orchestrator state
3. IF THE API Server is unreachable, THE Frontend Client SHALL display a clear error message
4. THE Orchestrator SHALL continue running and generating OSC messages even when no clients are connected
5. WHEN THE API Server shuts down, THE API Server SHALL gracefully stop the Orchestrator and close all WebSocket connections
