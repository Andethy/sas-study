import React, { useRef, useEffect, useState, useCallback } from 'react';

interface Tensions {
  zone1: number;
  zone2: number;
  zone3: number;
}

interface Point {
  x: number;
  y: number;
}

interface ZoneAnchor extends Point {
  color: string;
}

interface MidiControllerState {
  k1Knob: number;
  k2Knob: number;
  joystickX: number;
  joystickY: number;
}

interface TensionCanvasProps {
  tensions: Tensions;
  onTensionUpdate: (tensions: Tensions, source?: 'mouse' | 'midi') => void;
  isAutomationMode?: boolean;
  midiControllerState?: MidiControllerState | null;
  activeInputSource?: 'mouse' | 'midi';
  onMouseActiveChange?: (isActive: boolean) => void;
}

const TensionCanvas: React.FC<TensionCanvasProps> = ({ tensions, onTensionUpdate, isAutomationMode = false, midiControllerState = null, activeInputSource = 'mouse', onMouseActiveChange }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasSize, setCanvasSize] = useState<{ width: number; height: number }>({ width: 600, height: 600 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [canvasPoint, setCanvasPoint] = useState<Point>({ x: 300, y: 300 });

  // Update canvas point when canvas size changes to keep it centered
  useEffect(() => {
    const centerX = canvasSize.width / 2;
    const centerY = canvasSize.height / 2;
    console.log('Canvas size changed:', canvasSize, 'Setting point to:', { x: centerX, y: centerY });
    setCanvasPoint({ x: centerX, y: centerY });
  }, [canvasSize]);

  // Zone anchor positions (fixed)
  const zoneAnchors: Record<keyof Tensions, ZoneAnchor> = {
    zone1: { x: canvasSize.width * 0.1, y: canvasSize.height * 0.5, color: '#2196f3' }, // Left - Blue
    zone2: { x: canvasSize.width * 0.5, y: canvasSize.height * 0.1, color: '#4caf50' }, // Top - Green
    zone3: { x: canvasSize.width * 0.9, y: canvasSize.height * 0.5, color: '#f44336' }, // Right - Red
  };

  // Calculate distance between two points
  const calculateDistance = (p1: Point, p2: Point): number => {
    return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
  };

  // Calculate tension values based on canvas point position
  const calculateTensions = useCallback((point: Point): Tensions => {
    const maxDistance = canvasSize.width * 0.5; // Set to 0.5 for balanced overlap
    const newTensions: Tensions = { zone1: 0, zone2: 0, zone3: 0 };
    
    (Object.keys(zoneAnchors) as Array<keyof Tensions>).forEach(zone => {
      const anchor = zoneAnchors[zone];
      const distance = calculateDistance(point, anchor);
      
      // Inverse relationship: closer = higher tension
      // Use a smoother falloff curve
      let tension = Math.max(0, 1 - (distance / maxDistance));
      
      // Apply exponential falloff for more natural feel
      tension = Math.pow(tension, 1.5);
      
      // Set to 0 if too far away
      if (distance > maxDistance) {
        tension = 0;
      }
      
      newTensions[zone] = tension;
    });

    // Don't normalize - allow all zones to be zero when far away
    // Only apply a global scaling factor to keep values reasonable
    const globalScale = 1.0; // You can adjust this if needed
    
    (Object.keys(newTensions) as Array<keyof Tensions>).forEach(zone => {
      newTensions[zone] = newTensions[zone] * globalScale;
    });

    return newTensions;
  }, [canvasSize, zoneAnchors]);

  // Update tensions when canvas point moves (only if not in automation mode)
  useEffect(() => {
    if (isAutomationMode) {
      console.log('Skipping canvas tension update - automation mode active');
      return;
    }
    
    const newTensions = calculateTensions(canvasPoint);
    console.log('Canvas point moved to:', canvasPoint, 'New tensions:', newTensions);
    onTensionUpdate(newTensions, 'mouse');
  }, [canvasPoint, calculateTensions, isAutomationMode]);

  // Draw canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

    // Draw zone influence circles (colored by zone)
    const maxDistance = canvasSize.width * 0.5;
    ctx.lineWidth = 1;
    Object.entries(zoneAnchors).forEach(([zone, anchor]) => {
      // Convert hex color to rgba with low opacity
      const hexColor = anchor.color;
      const r = parseInt(hexColor.slice(1, 3), 16);
      const g = parseInt(hexColor.slice(3, 5), 16);
      const b = parseInt(hexColor.slice(5, 7), 16);
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.2)`;
      
      ctx.beginPath();
      ctx.arc(anchor.x, anchor.y, maxDistance, 0, 2 * Math.PI);
      ctx.stroke();
    });

    // Draw connection lines
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 2;
    Object.values(zoneAnchors).forEach(anchor => {
      ctx.beginPath();
      ctx.moveTo(canvasPoint.x, canvasPoint.y);
      ctx.lineTo(anchor.x, anchor.y);
      ctx.stroke();
    });

    // Draw zone anchors
    Object.entries(zoneAnchors).forEach(([zone, anchor]) => {
      ctx.fillStyle = anchor.color;
      ctx.beginPath();
      ctx.arc(anchor.x, anchor.y, 15, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw zone label
      ctx.fillStyle = '#fff';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(zone.toUpperCase(), anchor.x, anchor.y - 25);
      
      // Draw tension value
      const tension = tensions[zone as keyof Tensions] || 0;
      ctx.fillText(tension.toFixed(2), anchor.x, anchor.y + 35);
    });

    // Draw canvas point (draggable)
    // Draw mouse position with different styling based on active input
    ctx.fillStyle = activeInputSource === 'mouse' ? '#fff' : '#ccc';
    ctx.strokeStyle = activeInputSource === 'mouse' ? '#333' : '#666';
    ctx.lineWidth = activeInputSource === 'mouse' ? 3 : 2;
    ctx.beginPath();
    ctx.arc(canvasPoint.x, canvasPoint.y, activeInputSource === 'mouse' ? 12 : 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    
    // Draw MIDI controller position if available
    if (midiControllerState) {
      const midiX = ((midiControllerState.joystickX + 1) / 2) * canvasSize.width;
      const midiY = (1 - midiControllerState.joystickY) * canvasSize.height; // joystick Y: 0=bottom, 1=top
      
      // Draw MIDI position with different styling based on active input
      ctx.fillStyle = activeInputSource === 'midi' ? '#FFD700' : '#B8860B'; // Bright gold or dim gold
      ctx.strokeStyle = activeInputSource === 'midi' ? '#FF8C00' : '#8B6914'; // Bright or dim orange
      ctx.lineWidth = activeInputSource === 'midi' ? 3 : 2;
      ctx.beginPath();
      ctx.arc(midiX, midiY, activeInputSource === 'midi' ? 10 : 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      
      // Draw connection line between mouse and MIDI positions
      ctx.strokeStyle = '#FFD700';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(canvasPoint.x, canvasPoint.y);
      ctx.lineTo(midiX, midiY);
      ctx.stroke();
      ctx.setLineDash([]); // Reset line dash
    }
  }, [canvasSize, canvasPoint, zoneAnchors, tensions, midiControllerState, activeInputSource]);

  // Handle canvas resize
  useEffect(() => {
    const handleResize = () => {
      const container = canvasRef.current?.parentElement;
      if (container) {
        const rect = container.getBoundingClientRect();
        const size = Math.min(rect.width - 40, 600); // Account for padding
        console.log('Container rect:', rect, 'Setting canvas size to:', size);
        setCanvasSize({ width: size, height: size });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Draw when canvas updates
  useEffect(() => {
    draw();
  }, [draw]);

  // Get mouse/touch position relative to canvas
  const getEventPosition = (event: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>): Point => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvasSize.width / rect.width;
    const scaleY = canvasSize.height / rect.height;

    let clientX: number, clientY: number;
    if ('touches' in event && event.touches && event.touches[0]) {
      clientX = event.touches[0].clientX;
      clientY = event.touches[0].clientY;
    } else {
      clientX = (event as React.MouseEvent).clientX;
      clientY = (event as React.MouseEvent).clientY;
    }

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  };

  // Mouse/touch event handlers
  const handleStart = (event: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const pos = getEventPosition(event);
    const distance = calculateDistance(pos, canvasPoint);
    
    console.log('Mouse/touch start at:', pos, 'Distance from point:', distance);
    
    // Notify parent that mouse is now active
    onMouseActiveChange?.(true);
    
    if (distance <= 20) { // Within dragging range
      console.log('Starting drag');
      setIsDragging(true);
    }
  };

  const handleMove = (event: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;
    
    event.preventDefault();
    const pos = getEventPosition(event);
    
    // Constrain to canvas bounds
    const constrainedPos = {
      x: Math.max(20, Math.min(canvasSize.width - 20, pos.x)),
      y: Math.max(20, Math.min(canvasSize.height - 20, pos.y)),
    };
    
    console.log('Dragging to:', constrainedPos);
    setCanvasPoint(constrainedPos);
  };

  const handleEnd = (event: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    setIsDragging(false);
    
    // Notify parent that mouse is no longer active
    onMouseActiveChange?.(false);
  };

  // Click to move point
  const handleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) return;
    
    const pos = getEventPosition(event);
    const constrainedPos = {
      x: Math.max(20, Math.min(canvasSize.width - 20, pos.x)),
      y: Math.max(20, Math.min(canvasSize.height - 20, pos.y)),
    };
    
    console.log('Click to move point to:', constrainedPos);
    setCanvasPoint(constrainedPos);
  }; 
 return (
    <div className="canvas-container">
      <canvas
        ref={canvasRef}
        className="tension-canvas"
        width={canvasSize.width}
        height={canvasSize.height}
        onMouseDown={handleStart}
        onMouseMove={handleMove}
        onMouseUp={handleEnd}
        onMouseLeave={handleEnd}
        onTouchStart={handleStart}
        onTouchMove={handleMove}
        onTouchEnd={handleEnd}
        onClick={handleClick}
        style={{
          width: '100%',
          height: 'auto',
          maxWidth: '600px',
        }}
      />
    </div>
  );
};

export default TensionCanvas;