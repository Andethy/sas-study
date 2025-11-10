import { useState, useEffect, useRef, useCallback } from 'react';
import TensionCanvas from './components/TensionCanvas';
import Controls from './components/Controls';
import StatusLog from './components/StatusLog';
import { useWebSocket } from './hooks/useWebSocket';
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

  // Tension update handler with throttling
  const tensionUpdateTimeoutRef = useRef<number | null>(null);
  
  const handleTensionUpdate = useCallback((newTensions: Tensions) => {
    // console.log('Tension update received:', newTensions);
    // setTensions(newTensions);
    
    // Throttle API calls to avoid overwhelming the server
    if (tensionUpdateTimeoutRef.current) {
      // console.log('Clearing previous timeout:', tensionUpdateTimeoutRef.current);
      clearTimeout(tensionUpdateTimeoutRef.current);
    }
    
    const timeoutId = window.setTimeout(async () => {
      console.log('Timeout callback executing for tensions:', newTensions);
      try {
        console.log('Sending tension update to API:', newTensions);
        const result = await apiService.updateTension(newTensions);
        console.log('API response:', result);
        addLog(`Tensions updated: ${Object.entries(newTensions).map(([k,v]) => `${k}=${v.toFixed(2)}`).join(', ')}`, 'success');
      } catch (error) {
        console.error('Tension update error:', error);
        addLog(`Error updating tension: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
      }
    }, 100);
    
    tensionUpdateTimeoutRef.current = timeoutId;
    // console.log('Set new timeout:', timeoutId);
  }, [addLog]);

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

  const handleFillTrigger = async (preset: string, beats: number) => {
    try {
      await apiService.triggerFill(preset, beats);
      addLog(`Fill triggered: ${preset} (${beats} beats)`, 'success');
    } catch (error) {
      addLog(`Error triggering fill: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  };

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
        </div>
      </header>

      <div className="main-content">
        <div className="canvas-section">
          <TensionCanvas
            tensions={tensions}
            onTensionUpdate={handleTensionUpdate}
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
          </div>
        </div>

        <div className="controls-section">
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
          
          <StatusLog logs={logs} />
        </div>
      </div>
    </div>
  );
}

export default App;
