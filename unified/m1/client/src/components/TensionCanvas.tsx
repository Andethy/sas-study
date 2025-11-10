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

interface TensionCanvasProps {
  tensions: Tensions;
  onTensionUpdate: (tensions: Tensions) => void;
}

const TensionCanvas: React.FC<TensionCanvasProps> = ({ tensions, onTensionUpdate }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasSize, setCanvasSize] = useState<{ width: number; height: number }>({ width: 600, height: 600 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [canvasPoint, setCanvasPoint] = useState<Point>({ x: 300, y: 300 });

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
    const maxDistance = canvasSize.width * 0.4; // Reduced from 0.7 to 0.4 for tighter control
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

  // Update tensions when canvas point moves
  useEffect(() => {
    const newTensions = calculateTensions(canvasPoint);
    console.log('Canvas point moved to:', canvasPoint, 'New tensions:', newTensions);
    onTensionUpdate(newTensions);
  }, [canvasPoint, calculateTensions, onTensionUpdate]);

  // Draw canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

    // Draw zone influence circles (faint)
    const maxDistance = canvasSize.width * 0.4;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    Object.values(zoneAnchors).forEach(anchor => {
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
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(canvasPoint.x, canvasPoint.y, 12, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }, [canvasSize, canvasPoint, zoneAnchors, tensions]);

  // Handle canvas resize
  useEffect(() => {
    const handleResize = () => {
      const container = canvasRef.current?.parentElement;
      if (container) {
        const rect = container.getBoundingClientRect();
        const size = Math.min(rect.width, 600);
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