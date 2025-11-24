import React, { useState, useRef, useEffect } from 'react';
import { apiService } from '../services/apiService';

interface TimbreInterpolationProps {
  isConnected: boolean;
  onLog?: (message: string, type: 'info' | 'success' | 'error') => void;
}

const TimbreInterpolation: React.FC<TimbreInterpolationProps> = ({ isConnected, onLog }) => {
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [fileAUploaded, setFileAUploaded] = useState<string | null>(null);
  const [fileBUploaded, setFileBUploaded] = useState<string | null>(null);
  const [mixValue, setMixValue] = useState<number>(0.5);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isMixing, setIsMixing] = useState<boolean>(false);
  const [currentAudioUrl, setCurrentAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const fileInputARef = useRef<HTMLInputElement>(null);
  const fileInputBRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Load initial status
  useEffect(() => {
    if (isConnected) {
      loadTimbreStatus();
    }
  }, [isConnected]);

  const loadTimbreStatus = async () => {
    try {
      const status = await apiService.getTimbreStatus();
      setFileAUploaded(status.sample_a);
      setFileBUploaded(status.sample_b);
      setMixValue(status.current_mix);
    } catch (error) {
      console.error('Error loading timbre status:', error);
    }
  };

  const handleFileAChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && isConnected) {
      setFileA(file);
      await uploadFile(file, 'sample_a');
    }
  };

  const handleFileBChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && isConnected) {
      setFileB(file);
      await uploadFile(file, 'sample_b');
    }
  };

  const uploadFile = async (file: File, sampleType: 'sample_a' | 'sample_b') => {
    setIsUploading(true);
    try {
      const result = await apiService.uploadTimbreSample(file, sampleType);
      
      if (sampleType === 'sample_a') {
        setFileAUploaded(result.filename);
      } else {
        setFileBUploaded(result.filename);
      }
      
      onLog?.(`Uploaded ${sampleType}: ${result.filename}`, 'success');
    } catch (error) {
      onLog?.(`Error uploading ${sampleType}: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    } finally {
      setIsUploading(false);
    }
  };

  const handleMixChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const newMixValue = parseFloat(event.target.value);
    setMixValue(newMixValue);
    
    if (fileAUploaded && fileBUploaded && isConnected) {
      await performMix(newMixValue);
    }
  };

  const performMix = async (mixVal: number) => {
    setIsMixing(true);
    try {
      // Invert the mix value since the model expects:
      // r=0 -> 100% sample B, r=1 -> 100% sample A
      // But our slider shows: 0 -> A, 1 -> B
      const invertedMix = 1 - mixVal;
      const result = await apiService.setTimbreMix(invertedMix);
      onLog?.(`Timbre mix updated: ${(mixVal * 100).toFixed(0)}% B`, 'success');
      
      // If we got an audio URL, play the result automatically
      if (result.audio_url) {
        const fullAudioUrl = `http://localhost:8080${result.audio_url}`;
        setCurrentAudioUrl(fullAudioUrl);
        playAudio(fullAudioUrl);
      }
    } catch (error) {
      onLog?.(`Error mixing timbres: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    } finally {
      setIsMixing(false);
    }
  };

  const playAudio = (audioUrl: string) => {
    if (audioRef.current) {
      audioRef.current.src = audioUrl;
      audioRef.current.play()
        .then(() => {
          setIsPlaying(true);
          onLog?.('Playing interpolated audio', 'info');
        })
        .catch((error) => {
          console.error('Error playing audio:', error);
          onLog?.('Error playing audio', 'error');
        });
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  const handleAudioError = () => {
    setIsPlaying(false);
    onLog?.('Error loading audio file', 'error');
  };

  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
    }
  };

  const replayAudio = () => {
    if (currentAudioUrl) {
      playAudio(currentAudioUrl);
    }
  };

  const clearFileA = async () => {
    try {
      if (fileAUploaded) {
        await apiService.deleteTimbreSample('sample_a');
      }
      setFileA(null);
      setFileAUploaded(null);
      if (fileInputARef.current) {
        fileInputARef.current.value = '';
      }
      onLog?.('Sample A cleared', 'info');
    } catch (error) {
      onLog?.(`Error clearing sample A: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  };

  const clearFileB = async () => {
    try {
      if (fileBUploaded) {
        await apiService.deleteTimbreSample('sample_b');
      }
      setFileB(null);
      setFileBUploaded(null);
      if (fileInputBRef.current) {
        fileInputBRef.current.value = '';
      }
      onLog?.('Sample B cleared', 'info');
    } catch (error) {
      onLog?.(`Error clearing sample B: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    }
  };

  return (
    <div className="timbre-interpolation">
      <h3>Timbre Interpolation</h3>
      
      {/* File Upload Section */}
      <div className="file-upload-section">
        <div className="file-upload-group">
          <label className="file-upload-label">Sample A:</label>
          <div className="file-upload-controls">
            <input
              ref={fileInputARef}
              type="file"
              accept="audio/*"
              onChange={handleFileAChange}
              disabled={!isConnected}
              className="file-input"
            />
            {(fileA || fileAUploaded) && (
              <div className="file-info">
                <span className="file-name">{fileAUploaded || fileA?.name}</span>
                <button 
                  onClick={clearFileA}
                  className="clear-file-btn"
                  disabled={!isConnected || isUploading}
                >
                  ×
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="file-upload-group">
          <label className="file-upload-label">Sample B:</label>
          <div className="file-upload-controls">
            <input
              ref={fileInputBRef}
              type="file"
              accept="audio/*"
              onChange={handleFileBChange}
              disabled={!isConnected}
              className="file-input"
            />
            {(fileB || fileBUploaded) && (
              <div className="file-info">
                <span className="file-name">{fileBUploaded || fileB?.name}</span>
                <button 
                  onClick={clearFileB}
                  className="clear-file-btn"
                  disabled={!isConnected || isUploading}
                >
                  ×
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mix Slider Section */}
      <div className="mix-section">
        <label className="mix-label">Mix (A ← → B):</label>
        <div className="mix-controls">
          <span className="mix-indicator">A</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={mixValue}
            onChange={handleMixChange}
            disabled={!isConnected || !fileAUploaded || !fileBUploaded || isMixing}
            className="mix-slider"
          />
          <span className="mix-indicator">B</span>
        </div>
        <div className="mix-value">
          Mix: {(mixValue * 100).toFixed(0)}% B
        </div>
      </div>

      {/* Audio Controls */}
      {currentAudioUrl && (
        <div className="audio-controls">
          <button 
            onClick={replayAudio}
            disabled={!isConnected || isPlaying}
            className="audio-control-btn"
          >
            {isPlaying ? '🔊 Playing...' : '🔄 Replay'}
          </button>
          {isPlaying && (
            <button 
              onClick={stopAudio}
              className="audio-control-btn stop-btn"
            >
              ⏹ Stop
            </button>
          )}
        </div>
      )}

      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        onEnded={handleAudioEnded}
        onError={handleAudioError}
        style={{ display: 'none' }}
      />

      {/* Status */}
      <div className="timbre-status">
        {isUploading && (
          <p className="status-message">Uploading file...</p>
        )}
        {isMixing && (
          <p className="status-message">Processing timbre interpolation...</p>
        )}
        {!isUploading && !isMixing && !fileAUploaded && !fileBUploaded && (
          <p className="status-message">Upload two audio files to begin timbre interpolation</p>
        )}
        {!isUploading && !isMixing && fileAUploaded && !fileBUploaded && (
          <p className="status-message">Upload Sample B to enable mixing</p>
        )}
        {!isUploading && !isMixing && !fileAUploaded && fileBUploaded && (
          <p className="status-message">Upload Sample A to enable mixing</p>
        )}
        {!isUploading && !isMixing && fileAUploaded && fileBUploaded && (
          <p className="status-message status-ready">Ready for timbre interpolation - adjust slider to mix</p>
        )}
      </div>
    </div>
  );
};

export default TimbreInterpolation;