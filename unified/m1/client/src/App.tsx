import { useState, useEffect, useRef, useCallback } from 'react';
import TensionCanvas from './components/TensionCanvas';
import Controls from './components/Controls';
import StatusLog from './components/StatusLog';
import TimbreInterpolation from './components/TimbreInterpolation';
import TensionAutomation from './components/TensionAutomation';
import { useWebSocket } from './hooks/useWebSocket';
import { useMidiInput } from './hooks/useMidiInput';
import { apiService } from './services/apiService';

const WS_URL = import.meta.env.PROD
  ? `ws://${window.location.host}/ws`
  : 'ws://localhost:8080/ws';

interface Tensions {
  zone1: number;
  zone2: number;
  zone3: number;
}

interface OrchestratorState {
  bpm: number;
  beats_per_bar: number;
  key_root: string;
  current_bar: number;
}

interface LogEntry {
  id: number;
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'error';
}

function App() {
  // State
  const [tensions, setTensions] = useState<Tensions>({
    zone1: 0.2,
    zone2: 0.4,
    zone3: 0.7
  });
  
  const [orchestratorState, setOrchestratorState] = useState<OrchestratorState>({
    bpm: 120,
    beats_per_bar: 4,
    key_root: 'C',
    current_bar: 0
  });

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [activeTab, setActiveTab] = useState<'controls' | 'timbre' | 'automation'>('controls');
  const [lastMaxTension, setLastMaxTension] = useState<number>(0);
  const [isAutomationMode, setIsAutomationMode] = useState<boolean>(false);
  const [midiControlledTensions, setMidiControlledTensions] = useState<Tensions>({ zone1: 0.5, zone2: 0.5, zone3: 0.5 });
  const [inputSource, setInputSource] = useState<'mouse' | 'midi'>('mouse');
  const lastInputTimeRef = useRef<{ mouse: number; midi: number }>({ mouse: 0, midi: 0 });

  // WebSocket connection
  const { lastMessage, connectionStatus } = useWebSocket(WS_URL);

  // Utility function to add log entries
  const addLog = useCallback((message: string, type: LogEntry['type'] = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-49), { // Keep last 50 entries
      id: Date.now(),
      timestamp,
      message,
      type
    }]);
  }, []);

  // MIDI Input handlers
  const handleMidiTimbreVolumeChange = useCallback(async (volume: number) => {
    try {
      await apiService.setTimbreVolume(volume);
      addLog(`MIDI K2: Timbre volume set to ${(volume * 100).toFixed(0)}%`, 'info');
    } catch (error) {
      addLog(`MIDI Timbre Volume Error: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  }, [addLog]);

  const handleMidiTimbreMixChange = useCallback(async (mix: number) => {
    try {
      await apiService.setTimbreMix(mix);
      addLog(`MIDI K3: Timbre mix set to ${(mix * 100).toFixed(0)}% B`, 'info');
    } catch (error) {
      addLog(`MIDI Timbre Mix Error: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  }, [addLog]);

  const handleMidiMasterVolumeChange = useCallback(async (volume: number) => {
    try {
      await apiService.setMasterVolume(volume);
      addLog(`MIDI K4: Master volume set to ${(volume * 100).toFixed(0)}%`, 'info');
    } catch (error) {
      addLog(`MIDI Master Volume Error: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  }, [addLog]);

  // Create refs for functions that will be used in MIDI handlers
  const handleTensionUpdateRef = useRef<((tensions: Tensions) => void) | null>(null);
  
  const handleMidiJoystickChange = useCallback((x: number, y: number) => {
    // Map joystick to tension canvas with bottom-center origin
    // Joystick: x=[-1,1], y=[0,1] where y=0 is bottom, y=1 is top
    // Canvas: x=[0,1], y=[0,1] where y=0 is top, y=1 is bottom
    
    // Convert joystick coordinates to canvas coordinates
    const canvasX = (x + 1) / 2; // Convert -1,1 to 0,1
    const canvasY = 1 - y; // Convert 0,1 to 1,0 (flip Y axis: joystick 0=bottom -> canvas 1=bottom)
    
    // Calculate tensions based on distance from zones (same logic as TensionCanvas)
    const zones = [
      { x: 0.2, y: 0.5 },  // Zone 1 (left)
      { x: 0.5, y: 0.2 },  // Zone 2 (top)
      { x: 0.8, y: 0.5 }   // Zone 3 (right)
    ];
    
    const newTensions = zones.map((zone, index) => {
      const distance = Math.sqrt(
        Math.pow(canvasX - zone.x, 2) + Math.pow(canvasY - zone.y, 2)
      );
      const maxDistance = 0.5; // Half canvas width as influence radius
      const tension = Math.max(0, 1 - (distance / maxDistance));
      return tension;
    });
    
    const tensionUpdate = {
      zone1: newTensions[0],
      zone2: newTensions[1],
      zone3: newTensions[2]
    };
    
    setMidiControlledTensions(tensionUpdate);
    
    // Track MIDI input timing and set as active input source
    const now = Date.now();
    lastInputTimeRef.current.midi = now;
    setInputSource('midi');
    
    // Use ref to call handleTensionUpdate if available
    if (!isAutomationMode && handleTensionUpdateRef.current) {
      handleTensionUpdateRef.current(tensionUpdate, 'midi');
    }
  }, [isAutomationMode]);

  // Update connection status
  useEffect(() => {
    console.log('Connection status changed:', connectionStatus);
    setIsConnected(connectionStatus === 'Connected');
    
    // Test API connection when WebSocket connects
    if (connectionStatus === 'Connected') {
      console.log('Testing API connection...');
      apiService.getState()
        .then(state => {
          console.log('API test successful:', state);
          addLog('API connection verified', 'success');
        })
        .catch(error => {
          console.error('API test failed:', error);
          addLog(`API connection failed: ${error.message}`, 'error');
        });
    }
  }, [connectionStatus, addLog]);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data);
      
      switch (data.type) {
        case 'initial_state':
          setOrchestratorState(data.state);
          setTensions(data.state.tensions);
          addLog('Connected to orchestrator', 'success');
          break;
          
        case 'tension_update':
          setTensions(data.tensions);
          break;
          
        case 'bpm_update':
          setOrchestratorState(prev => ({ ...prev, bpm: data.bpm }));
          addLog(`BPM updated to ${data.bpm}`, 'info');
          break;
          
        case 'key_update':
          setOrchestratorState(prev => ({ ...prev, key_root: data.key }));
          addLog(`Key updated to ${data.key}`, 'info');
          break;
          
        case 'beats_per_bar_update':
          setOrchestratorState(prev => ({ ...prev, beats_per_bar: data.beats_per_bar }));
          addLog(`Beats per bar updated to ${data.beats_per_bar}`, 'info');
          break;
          
        case 'bar_update':
          setOrchestratorState(prev => ({ ...prev, current_bar: data.bar }));
          break;
          
        case 'chord_queued':
          addLog(`Chord queued: ${data.chord}`, 'success');
          break;
          
        case 'fill_triggered':
          addLog(`Fill triggered: ${data.preset} (${data.beats} beats)`, 'success');
          break;
          
        default:
          console.log('Unknown message type:', data.type);
      }
    }
  }, [lastMessage]);

  // Control handlers (define these first)
  const handleFillTrigger = useCallback(async (preset: string, beats: number) => {
    try {
      await apiService.triggerFill(preset, beats);
      addLog(`Fill triggered: ${preset} (${beats} beats)`, 'success');
    } catch (error) {
      addLog(`Error triggering fill: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  }, [addLog]);

  // Tension update handler with throttling
  const tensionUpdateTimeoutRef = useRef<number | null>(null);
  
  const handleTensionUpdate = useCallback((newTensions: Tensions, source: 'mouse' | 'midi' = 'mouse') => {
    console.log('Tension update received:', newTensions, 'from:', source);
    
    // Track input source and timing
    const now = Date.now();
    lastInputTimeRef.current[source] = now;
    
    // Check if this input should be ignored due to recent input from another source
    const otherSource = source === 'mouse' ? 'midi' : 'mouse';
    const timeSinceOtherInput = now - lastInputTimeRef.current[otherSource];
    
    // If the other input source was active within the last 200ms, ignore this update
    if (timeSinceOtherInput < 200 && inputSource !== source) {
      console.log(`Ignoring ${source} input due to recent ${otherSource} activity`);
      return;
    }
    
    // Update active input source
    setInputSource(source);
    
    // Check if tensions actually changed to prevent unnecessary updates
    const currentTensions = tensions;
    const hasChanged = Object.keys(newTensions).some(key => 
      Math.abs(newTensions[key as keyof Tensions] - (currentTensions[key as keyof Tensions] || 0)) > 0.001
    );
    
    if (!hasChanged) {
      console.log('Tensions unchanged, skipping update');
      return;
    }
    
    setTensions(newTensions);
    
    // Check for automatic fill trigger
    const maxTension = Math.max(newTensions.zone1, newTensions.zone2, newTensions.zone3);
    if (maxTension > 0.85 && lastMaxTension <= 0.85) {
      console.log('Max tension exceeded 0.85, triggering automatic fill');
      // Call the fill trigger directly to avoid circular dependency
      apiService.triggerFill('snare', 2).then(() => {
        addLog('Automatic fill triggered (high tension)', 'info');
      }).catch((error) => {
        addLog(`Error triggering automatic fill: ${error.message}`, 'error');
      });
    }
    setLastMaxTension(maxTension);
    
    // Throttle API calls to avoid overwhelming the server
    if (tensionUpdateTimeoutRef.current) {
      console.log('Clearing previous timeout:', tensionUpdateTimeoutRef.current);
      clearTimeout(tensionUpdateTimeoutRef.current);
    }
    
    const timeoutId = window.setTimeout(async () => {
      console.log('🚀 Timeout callback executing for tensions:', newTensions);
      try {
        console.log('📤 Sending tension update to API:', newTensions);
        const result = await apiService.updateTension(newTensions);
        console.log('✅ API response:', result);
        addLog(`Tensions updated: ${Object.entries(newTensions).map(([k,v]) => `${k}=${v.toFixed(2)}`).join(', ')}`, 'success');
      } catch (error) {
        console.error('❌ Tension update error:', error);
        addLog(`Error updating tension: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
      }
    }, 500); // Increased from 100ms to 500ms to allow API calls to complete
    
    tensionUpdateTimeoutRef.current = timeoutId;
    console.log('Set new timeout:', timeoutId);
  }, [addLog, lastMaxTension]);

  // Update the ref so MIDI handlers can use it
  useEffect(() => {
    handleTensionUpdateRef.current = handleTensionUpdate;
  }, [handleTensionUpdate]);

  // Control handlers
  const handleBPMUpdate = async (bpm: number) => {
    try {
      await apiService.updateBPM(bpm);
      addLog(`BPM set to ${bpm}`, 'success');
    } catch (error) {
      addLog(`Error setting BPM: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  };

  const handleKeyUpdate = async (key: string) => {
    try {
      await apiService.updateKey(key);
      addLog(`Key set to ${key}`, 'success');
    } catch (error) {
      addLog(`Error setting key: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  };

  const handleBeatsPerBarUpdate = async (beatsPerBar: number) => {
    try {
      await apiService.updateBeatsPerBar(beatsPerBar);
      addLog(`Beats per bar set to ${beatsPerBar}`, 'success');
    } catch (error) {
      addLog(`Error setting beats per bar: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  };

  const handleChordQueue = async (chord: string) => {
    try {
      await apiService.queueChord(chord);
      addLog(`Chord queued: ${chord}`, 'success');
    } catch (error) {
      addLog(`Error queuing chord: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  };



  // Keyboard event listener for spacebar drum fill
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      // Only trigger if spacebar is pressed and not in an input field
      if (event.code === 'Space' && 
          event.target instanceof HTMLElement && 
          !['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) {
        event.preventDefault();
        console.log('Spacebar pressed - triggering drum fill');
        handleFillTrigger('snare', 3); // Default: snare fill, 3 beats
        addLog('Drum fill triggered via spacebar', 'info');
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handleFillTrigger, addLog]);

  // MIDI connection
  const { isConnected: midiConnected, deviceName, controllerState } = useMidiInput({
    onTimbreVolumeChange: handleMidiTimbreVolumeChange,
    onTimbreMixChange: handleMidiTimbreMixChange,
    onMasterVolumeChange: handleMidiMasterVolumeChange,
    onJoystickChange: handleMidiJoystickChange,
    onLog: addLog
  });

  return (
    <div className="container">
      <header>
        <h1>Orchestrator</h1>
        <div className="connection-status">
          <span 
            className={`status-indicator ${isConnected ? 'status-connected' : 'status-disconnected'}`}
          />
          <span className="status-text">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
          {midiConnected && (
            <div className="midi-status">
              <span className="midi-device">{deviceName || 'MIDI Connected'}</span>
              <div className="midi-values">
                <span>K2-TVol: {(controllerState.k2Knob * 100).toFixed(0)}%</span>
                <span>K3-TMix: {(controllerState.k3Knob * 100).toFixed(0)}%</span>
                <span>K4-MVol: {(controllerState.k4Knob * 100).toFixed(0)}%</span>
                <span>Joy: ({controllerState.joystickX.toFixed(2)}, {controllerState.joystickY.toFixed(2)})</span>
              </div>
            </div>
          )}
        </div>
      </header>

      <div className="main-content">
        <div className="canvas-section">
          <TensionCanvas
            tensions={tensions}
            onTensionUpdate={handleTensionUpdate}
            isAutomationMode={isAutomationMode}
            midiControllerState={midiConnected ? controllerState : null}
            activeInputSource={inputSource}
          />
          
          <div className="tension-display">
            <div className="tension-value">
              <span className="zone-label zone1-color">Zone 1:</span>
              <span>{tensions.zone1.toFixed(2)}</span>
            </div>
            <div className="tension-value">
              <span className="zone-label zone2-color">Zone 2:</span>
              <span>{tensions.zone2.toFixed(2)}</span>
            </div>
            <div className="tension-value">
              <span className="zone-label zone3-color">Zone 3:</span>
              <span>{tensions.zone3.toFixed(2)}</span>
            </div>
            {isAutomationMode && (
              <div className="automation-indicator">
                <span style={{ color: '#ff9800', fontWeight: 'bold' }}>🤖 AUTOMATION ACTIVE</span>
              </div>
            )}
          </div>
        </div>

        <div className="controls-section">
          {/* Tabs */}
          <div className="tabs-container">
            <div className="tabs-header">
              <button 
                className={`tab-button ${activeTab === 'controls' ? 'active' : ''}`}
                onClick={() => setActiveTab('controls')}
              >
                Controls
              </button>
              <button 
                className={`tab-button ${activeTab === 'timbre' ? 'active' : ''}`}
                onClick={() => setActiveTab('timbre')}
              >
                Timbre
              </button>
              <button 
                className={`tab-button ${activeTab === 'automation' ? 'active' : ''}`}
                onClick={() => setActiveTab('automation')}
              >
                Automation
              </button>
            </div>
            
            <div className="tab-content">
              {activeTab === 'controls' && (
                <>
                  {/* Debug button to test tension updates */}
                  <div className="control-group">
                    <h3>Debug</h3>
                    <button 
                      onClick={() => {
                        console.log('Manual tension test button clicked');
                        handleTensionUpdate({ zone1: 0.1, zone2: 0.5, zone3: 0.9 });
                      }}
                      className="action-btn"
                    >
                      Test Tension Update
                    </button>
                    <button 
                      onClick={async () => {
                        console.log('Direct API test button clicked');
                        try {
                          const result = await apiService.updateTension({ zone1: 0.3, zone2: 0.6, zone3: 0.8 });
                          console.log('✅ Direct API test successful:', result);
                          addLog('Direct API test successful', 'success');
                        } catch (error) {
                          console.error('❌ Direct API test failed:', error);
                          addLog(`Direct API test failed: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
                        }
                      }}
                      className="action-btn"
                      style={{ marginTop: '10px' }}
                    >
                      Test Direct API
                    </button>
                  </div>
                  
                  <Controls
                    orchestratorState={orchestratorState}
                    onBPMUpdate={handleBPMUpdate}
                    onKeyUpdate={handleKeyUpdate}
                    onBeatsPerBarUpdate={handleBeatsPerBarUpdate}
                    onChordQueue={handleChordQueue}
                    onFillTrigger={handleFillTrigger}
                    isConnected={isConnected}
                  />
                </>
              )}
              
              {activeTab === 'timbre' && (
                <TimbreInterpolation 
                  isConnected={isConnected} 
                  onLog={addLog}
                />
              )}
              
              {activeTab === 'automation' && (
                <TensionAutomation 
                  isConnected={isConnected} 
                  onTensionUpdate={handleTensionUpdate}
                  onAutomationModeChange={setIsAutomationMode}
                />
              )}
            </div>
          </div>
          
          <StatusLog logs={logs} />
        </div>
      </div>
    </div>
  );
}

export default App;
