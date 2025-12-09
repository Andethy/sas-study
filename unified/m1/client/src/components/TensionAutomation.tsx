import React, { useState, useRef, useEffect, useCallback } from 'react';

interface Point {
  x: number; // Time (0-1, representing 0-100% of duration)
  y: number; // Tension (0-1)
}

interface TensionAutomationProps {
  isConnected: boolean;
  onTensionUpdate: (tensions: { zone1: number; zone2: number; zone3: number }, source?: 'mouse' | 'midi') => void;
  onAutomationModeChange: (isActive: boolean) => void;
}

const TensionAutomation: React.FC<TensionAutomationProps> = ({ isConnected, onTensionUpdate, onAutomationModeChange }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<Point[]>([
    { x: 0, y: 0.5 },
    { x: 1, y: 0.5 }
  ]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(10); // seconds
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackInterval, setPlaybackInterval] = useState<number | null>(null);
  const [draggedPointIndex, setDraggedPointIndex] = useState<number | null>(null);
  
  const canvasWidth = 600;
  const canvasHeight = 300;
  const padding = 40;

  // Convert canvas coordinates to curve coordinates
  const canvasToPoint = (canvasX: number, canvasY: number): Point => {
    const x = Math.max(0, Math.min(1, (canvasX - padding) / (canvasWidth - 2 * padding)));
    const y = Math.max(0, Math.min(1, 1 - (canvasY - padding) / (canvasHeight - 2 * padding)));
    
    console.log('Canvas click:', { canvasX, canvasY, x, y, padding, canvasWidth, canvasHeight }); // Debug
    return { x, y };
  };

  // Convert curve coordinates to canvas coordinates
  const pointToCanvas = (point: Point) => {
    const canvasX = padding + point.x * (canvasWidth - 2 * padding);
    const canvasY = padding + (1 - point.y) * (canvasHeight - 2 * padding);
    return { x: canvasX, y: canvasY };
  };

  // Interpolate tension value at given time
  const getTensionAtTime = useCallback((time: number): number => {
    if (points.length === 0) return 0;
    if (points.length === 1) return points[0].y;

    // Sort points by x coordinate
    const sortedPoints = [...points].sort((a, b) => a.x - b.x);
    
    // Find the two points to interpolate between
    let leftPoint = sortedPoints[0];
    let rightPoint = sortedPoints[sortedPoints.length - 1];
    
    for (let i = 0; i < sortedPoints.length - 1; i++) {
      if (time >= sortedPoints[i].x && time <= sortedPoints[i + 1].x) {
        leftPoint = sortedPoints[i];
        rightPoint = sortedPoints[i + 1];
        break;
      }
    }
    
    // Linear interpolation
    if (leftPoint.x === rightPoint.x) return leftPoint.y;
    const t = (time - leftPoint.x) / (rightPoint.x - leftPoint.x);
    return leftPoint.y + t * (rightPoint.y - leftPoint.y);
  }, [points]);

  // Draw the curve
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    
    // Vertical grid lines (time)
    for (let i = 0; i <= 10; i++) {
      const x = padding + (i / 10) * (canvasWidth - 2 * padding);
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, canvasHeight - padding);
      ctx.stroke();
    }
    
    // Horizontal grid lines (tension)
    for (let i = 0; i <= 10; i++) {
      const y = padding + (i / 10) * (canvasHeight - 2 * padding);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(canvasWidth - padding, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding, canvasHeight - padding);
    ctx.lineTo(canvasWidth - padding, canvasHeight - padding);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvasHeight - padding);
    ctx.stroke();

    // Draw labels
    ctx.fillStyle = '#aaa';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    // Time labels
    for (let i = 0; i <= 10; i++) {
      const x = padding + (i / 10) * (canvasWidth - 2 * padding);
      const time = (i / 10) * duration;
      ctx.fillText(`${time.toFixed(1)}s`, x, canvasHeight - 10);
    }
    
    // Tension labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 10; i++) {
      const y = padding + (i / 10) * (canvasHeight - 2 * padding);
      const tension = 1 - (i / 10);
      ctx.fillText(tension.toFixed(1), padding - 10, y + 4);
    }

    // Draw curve
    if (points.length > 1) {
      const sortedPoints = [...points].sort((a, b) => a.x - b.x);
      
      ctx.strokeStyle = '#2196f3';
      ctx.lineWidth = 3;
      ctx.beginPath();
      
      const firstCanvas = pointToCanvas(sortedPoints[0]);
      ctx.moveTo(firstCanvas.x, firstCanvas.y);
      
      for (let i = 1; i < sortedPoints.length; i++) {
        const canvasPoint = pointToCanvas(sortedPoints[i]);
        ctx.lineTo(canvasPoint.x, canvasPoint.y);
      }
      
      ctx.stroke();
    }

    // Draw points
    points.forEach(point => {
      const canvasPoint = pointToCanvas(point);
      ctx.fillStyle = '#2196f3';
      ctx.beginPath();
      ctx.arc(canvasPoint.x, canvasPoint.y, 6, 0, 2 * Math.PI);
      ctx.fill();
      
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Draw playback position
    if (isPlaying) {
      const playbackX = padding + (currentTime / duration) * (canvasWidth - 2 * padding);
      ctx.strokeStyle = '#f44336';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playbackX, padding);
      ctx.lineTo(playbackX, canvasHeight - padding);
      ctx.stroke();
    }
  }, [points, isPlaying, currentTime, duration]);

  // Redraw when dependencies change
  useEffect(() => {
    draw();
  }, [draw]);

  // Handle mouse events
  const handleMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (isPlaying) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    console.log('Mouse down at:', { mouseX, mouseY, rectWidth: rect.width, rectHeight: rect.height }); // Debug log
    
    // Check if clicking near an existing point
    const clickedPointIndex = points.findIndex(point => {
      const canvasPoint = pointToCanvas(point);
      const distance = Math.sqrt(
        Math.pow(mouseX - canvasPoint.x, 2) + Math.pow(mouseY - canvasPoint.y, 2)
      );
      return distance <= 15; // Increased hit area
    });
    
    if (clickedPointIndex !== -1) {
      // Start dragging existing point
      console.log('Dragging point:', clickedPointIndex);
      setIsDrawing(true);
      setDraggedPointIndex(clickedPointIndex);
    } else {
      // Add new point
      const newPoint = canvasToPoint(mouseX, mouseY);
      console.log('Adding new point:', newPoint);
      setPoints(prev => [...prev, newPoint]);
    }
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || isPlaying || draggedPointIndex === null) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    const newPoint = canvasToPoint(mouseX, mouseY);
    
    console.log('Moving point to:', newPoint); // Debug log
    
    // Update the specific dragged point
    setPoints(prev => prev.map((point, index) => 
      index === draggedPointIndex ? newPoint : point
    ));
  };

  const handleMouseUp = () => {
    console.log('Mouse up - stopping drag'); // Debug log
    setIsDrawing(false);
    setDraggedPointIndex(null);
  };

  // Playback controls
  const startPlayback = () => {
    if (!isConnected) return;
    
    setIsPlaying(true);
    setCurrentTime(0);
    onAutomationModeChange(true); // Enable automation mode
    
    const interval = window.setInterval(() => {
      setCurrentTime(prevTime => {
        const newTime = prevTime + 0.1; // Update every 100ms
        
        if (newTime >= duration) {
          setIsPlaying(false);
          clearInterval(interval);
          setPlaybackInterval(null);
          onAutomationModeChange(false); // Disable automation mode
          return 0;
        }
        
        // Send tension update
        const normalizedTime = newTime / duration;
        const tensionValue = getTensionAtTime(normalizedTime);
        onTensionUpdate({
          zone1: tensionValue,
          zone2: tensionValue,
          zone3: tensionValue
        }, 'mouse'); // Automation is considered mouse input
        
        return newTime;
      });
    }, 100);
    
    setPlaybackInterval(interval);
  };

  const stopPlayback = () => {
    setIsPlaying(false);
    setCurrentTime(0);
    onAutomationModeChange(false); // Disable automation mode
    if (playbackInterval) {
      clearInterval(playbackInterval);
      setPlaybackInterval(null);
    }
  };

  const clearCurve = () => {
    if (isPlaying) return;
    console.log('Clearing curve'); // Debug log
    setPoints([
      { x: 0, y: 0.5 },
      { x: 1, y: 0.5 }
    ]);
  };

  return (
    <div className="tension-automation">
      <h3>Tension Automation</h3>
      
      <div className="automation-controls">
        <div className="control-row">
          <label htmlFor="duration-input">Duration:</label>
          <input
            id="duration-input"
            type="number"
            min="1"
            max="60"
            value={duration}
            onChange={(e) => setDuration(Math.max(1, parseInt(e.target.value)))}
            disabled={isPlaying}
          />
          <span>seconds</span>
        </div>
        
        <div className="playback-controls">
          <button
            onClick={startPlayback}
            disabled={isPlaying}
            className="play-btn"
          >
            ▶ Play
          </button>
          <button
            onClick={stopPlayback}
            disabled={!isPlaying}
            className="stop-btn"
          >
            ⏹ Stop
          </button>
          <button
            onClick={clearCurve}
            disabled={isPlaying}
            className="clear-btn"
          >
            Clear
          </button>
        </div>
        
        {isPlaying && (
          <div className="playback-status">
            Playing: {currentTime.toFixed(1)}s / {duration}s
          </div>
        )}
      </div>

      <div className="curve-editor">
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={canvasHeight}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="automation-canvas"
        />
        <p className="canvas-instructions">
          Click to add points, drag to move them. The curve represents tension over time.
        </p>
      </div>
    </div>
  );
};

export default TensionAutomation;