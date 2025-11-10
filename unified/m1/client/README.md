# Orchestrator Frontend

React + TypeScript + Vite frontend for the realtime orchestrator control system.

## Features

- **2D Canvas Interface**: Drag a point on the canvas to control tension zones
- **Zone-based Tension Control**: Three zones (left, top, right) with distance-based tension calculation
- **Real-time WebSocket Updates**: Live feedback from the orchestrator backend
- **Global Controls**: BPM, key signature, beats per bar
- **Performance Controls**: Drum fills, chord overrides
- **Status Logging**: Real-time feedback and error reporting
- **TypeScript**: Full type safety and IntelliSense support

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will run on http://localhost:5173 (Vite default) and connect to the API server at http://localhost:8080.

## Usage

### Canvas Control
- **Drag the white point** on the canvas to control tension zones
- **Zone anchors** are fixed at:
  - Zone 1 (Blue): Left side
  - Zone 2 (Green): Top
  - Zone 3 (Red): Right side
- **Tension values** are calculated inversely proportional to distance
- **Connection lines** show the relationship between your position and each zone

### Global Controls
- **BPM**: Set the tempo (40-240 BPM)
- **Key**: Choose the musical key (C, C#, D, etc.)
- **Beats/Bar**: Set the time signature (1-12 beats per bar)

### Performance Controls
- **Drum Fills**: Trigger fills with different presets (snare, toms, hats)
- **Chord Override**: Queue specific chords for the next bar (format: "C_M", "F_M7")

### Status
- **Connection indicator**: Shows WebSocket connection status
- **Current state**: Displays current bar, BPM, key, and beats per bar
- **Status log**: Real-time feedback and error messages

## API Integration

The frontend communicates with the orchestrator API via:
- **HTTP REST endpoints** for control operations
- **WebSocket connection** for real-time state updates

Make sure the API server is running on port 8080 before starting the frontend.