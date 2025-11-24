import React, { useState, useRef } from 'react';

interface TimbreInterpolationProps {
  isConnected: boolean;
}

const TimbreInterpolation: React.FC<TimbreInterpolationProps> = ({ isConnected }) => {
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [mixValue, setMixValue] = useState<number>(0.5);
  const fileInputARef = useRef<HTMLInputElement>(null);
  const fileInputBRef = useRef<HTMLInputElement>(null);

  const handleFileAChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setFileA(file);
    }
  };

  const handleFileBChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setFileB(file);
    }
  };

  const handleMixChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setMixValue(parseFloat(event.target.value));
  };

  const clearFileA = () => {
    setFileA(null);
    if (fileInputARef.current) {
      fileInputARef.current.value = '';
    }
  };

  const clearFileB = () => {
    setFileB(null);
    if (fileInputBRef.current) {
      fileInputBRef.current.value = '';
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
            {fileA && (
              <div className="file-info">
                <span className="file-name">{fileA.name}</span>
                <button 
                  onClick={clearFileA}
                  className="clear-file-btn"
                  disabled={!isConnected}
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
            {fileB && (
              <div className="file-info">
                <span className="file-name">{fileB.name}</span>
                <button 
                  onClick={clearFileB}
                  className="clear-file-btn"
                  disabled={!isConnected}
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
            disabled={!isConnected || !fileA || !fileB}
            className="mix-slider"
          />
          <span className="mix-indicator">B</span>
        </div>
        <div className="mix-value">
          Mix: {(mixValue * 100).toFixed(0)}% B
        </div>
      </div>

      {/* Status */}
      <div className="timbre-status">
        {!fileA && !fileB && (
          <p className="status-message">Upload two audio files to begin timbre interpolation</p>
        )}
        {fileA && !fileB && (
          <p className="status-message">Upload Sample B to enable mixing</p>
        )}
        {!fileA && fileB && (
          <p className="status-message">Upload Sample A to enable mixing</p>
        )}
        {fileA && fileB && (
          <p className="status-message status-ready">Ready for timbre interpolation</p>
        )}
      </div>
    </div>
  );
};

export default TimbreInterpolation;