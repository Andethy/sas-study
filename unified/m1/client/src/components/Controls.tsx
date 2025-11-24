import React, { useState } from 'react';

interface OrchestratorState {
  bpm: number;
  beats_per_bar: number;
  key_root: string;
  current_bar: number;
}

interface ControlsProps {
  orchestratorState: OrchestratorState;
  onBPMUpdate: (bpm: number) => void;
  onKeyUpdate: (key: string) => void;
  onBeatsPerBarUpdate: (beatsPerBar: number) => void;
  onChordQueue: (chord: string) => void;
  onFillTrigger: (preset: string, beats: number) => void;
  isConnected: boolean;
}

const Controls: React.FC<ControlsProps> = ({
  orchestratorState,
  onBPMUpdate,
  onKeyUpdate,
  onBeatsPerBarUpdate,
  onChordQueue,
  onFillTrigger,
  isConnected
}) => {
  const [bpm, setBpm] = useState<number>(orchestratorState.bpm);
  const [key, setKey] = useState<string>(orchestratorState.key_root);
  const [beatsPerBar, setBeatsPerBar] = useState<number>(orchestratorState.beats_per_bar);
  const [fillPreset, setFillPreset] = useState<string>('snare');
  const [fillBeats, setFillBeats] = useState<number>(3);
  const [chordInput, setChordInput] = useState<string>('C_M');

  const keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const fillPresets = ['snare', 'toms', 'hats'];

  // Update local state when orchestrator state changes
  React.useEffect(() => {
    setBpm(orchestratorState.bpm);
    setKey(orchestratorState.key_root);
    setBeatsPerBar(orchestratorState.beats_per_bar);
  }, [orchestratorState]);

  const handleBPMSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const bpmValue = Math.max(40, Math.min(240, parseFloat(bpm.toString())));
    setBpm(bpmValue);
    onBPMUpdate(bpmValue);
  };

  const handleKeySubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onKeyUpdate(key);
  };

  const handleBeatsPerBarSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const bpbValue = Math.max(1, Math.min(12, parseInt(beatsPerBar.toString())));
    setBeatsPerBar(bpbValue);
    onBeatsPerBarUpdate(bpbValue);
  };

  const handleFillTrigger = (e: React.FormEvent) => {
    e.preventDefault();
    const beatsValue = Math.max(1, Math.min(32, parseInt(fillBeats.toString())));
    setFillBeats(beatsValue);
    onFillTrigger(fillPreset, beatsValue);
  };

  const handleChordQueue = (e: React.FormEvent) => {
    e.preventDefault();
    if (chordInput.includes('_')) {
      onChordQueue(chordInput);
    }
  }; 
 return (
    <>
      {/* Global Controls */}
      <div className="control-group">
        <h3>Global Settings</h3>
        
        <form onSubmit={handleBPMSubmit}>
          <div className="control-row">
            <label htmlFor="bpm-input">BPM:</label>
            <input
              id="bpm-input"
              type="number"
              min="40"
              max="240"
              value={bpm}
              onChange={(e) => setBpm(parseFloat(e.target.value))}
              disabled={!isConnected}
            />
            <button type="submit" disabled={!isConnected}>
              Set
            </button>
          </div>
        </form>

        <form onSubmit={handleKeySubmit}>
          <div className="control-row">
            <label htmlFor="key-select">Key:</label>
            <select
              id="key-select"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              disabled={!isConnected}
            >
              {keys.map(k => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
            <button type="submit" disabled={!isConnected}>
              Set
            </button>
          </div>
        </form>

        <form onSubmit={handleBeatsPerBarSubmit}>
          <div className="control-row">
            <label htmlFor="bpb-input">Beats/Bar:</label>
            <input
              id="bpb-input"
              type="number"
              min="1"
              max="12"
              value={beatsPerBar}
              onChange={(e) => setBeatsPerBar(parseInt(e.target.value))}
              disabled={!isConnected}
            />
            <button type="submit" disabled={!isConnected}>
              Set
            </button>
          </div>
        </form>
      </div>

      {/* Drum Fills */}
      <div className="control-group">
        <h3>Drum Fills</h3>
        
        <form onSubmit={handleFillTrigger}>
          <div className="control-row">
            <label htmlFor="fill-preset">Preset:</label>
            <select
              id="fill-preset"
              value={fillPreset}
              onChange={(e) => setFillPreset(e.target.value)}
              disabled={!isConnected}
            >
              {fillPresets.map(preset => (
                <option key={preset} value={preset}>
                  {preset.charAt(0).toUpperCase() + preset.slice(1)}
                </option>
              ))}
            </select>
          </div>
          
          <div className="control-row">
            <label htmlFor="fill-beats">Beats:</label>
            <input
              id="fill-beats"
              type="number"
              min="1"
              max="32"
              value={fillBeats}
              onChange={(e) => setFillBeats(parseInt(e.target.value))}
              disabled={!isConnected}
            />
          </div>
          
          <button type="submit" className="action-btn" disabled={!isConnected}>
            Trigger Fill
          </button>
          <div style={{ fontSize: '12px', color: '#aaa', marginTop: '8px', textAlign: 'center' }}>
            Press <kbd style={{ background: '#333', padding: '2px 6px', borderRadius: '3px', fontSize: '11px' }}>Space</kbd> for quick snare fill
          </div>
        </form>
      </div>  
    {/* Chord Override */}
      <div className="control-group">
        <h3>Chord Override</h3>
        
        <form onSubmit={handleChordQueue}>
          <div className="control-row">
            <label htmlFor="chord-input">Chord:</label>
            <input
              id="chord-input"
              type="text"
              placeholder="C_M, F_M7"
              value={chordInput}
              onChange={(e) => setChordInput(e.target.value)}
              disabled={!isConnected}
            />
          </div>
          
          <button type="submit" className="action-btn" disabled={!isConnected}>
            Queue Chord
          </button>
        </form>
      </div>

      {/* Current State Display */}
      <div className="control-group">
        <h3>Current State</h3>
        <div className="control-row">
          <span>Bar: {orchestratorState.current_bar}</span>
        </div>
        <div className="control-row">
          <span>BPM: {orchestratorState.bpm}</span>
        </div>
        <div className="control-row">
          <span>Key: {orchestratorState.key_root}</span>
        </div>
        <div className="control-row">
          <span>Beats/Bar: {orchestratorState.beats_per_bar}</span>
        </div>
      </div>
    </>
  );
};

export default Controls;