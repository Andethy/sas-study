import { useState, useEffect, useRef, useCallback } from 'react';

type ConnectionStatus = 'Connecting' | 'Connected' | 'Disconnected' | 'Reconnecting' | 'Error';

interface UseWebSocketReturn {
  socket: WebSocket | null;
  lastMessage: MessageEvent | null;
  connectionStatus: ConnectionStatus;
  sendMessage: (message: any) => void;
}

export const useWebSocket = (url: string): UseWebSocketReturn => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [lastMessage, setLastMessage] = useState<MessageEvent | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('Connecting');
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const maxReconnectAttempts = 10;

  const connect = useCallback(() => {
    try {
      console.log('Attempting to connect to WebSocket:', url);
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        console.log('WebSocket connected successfully');
        setConnectionStatus('Connected');
        setSocket(ws);
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event: MessageEvent) => {
        console.log('WebSocket message received:', event.data);
        setLastMessage(event);
      };

      ws.onclose = (event: CloseEvent) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setConnectionStatus('Disconnected');
        setSocket(null);
        
        // Attempt to reconnect if not a manual close
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1})`);
          
          reconnectTimeoutRef.current = window.setTimeout(() => {
            reconnectAttemptsRef.current++;
            setConnectionStatus('Reconnecting');
            connect();
          }, delay);
        }
      };

      ws.onerror = (error: Event) => {
        console.error('WebSocket error occurred:', error);
        console.error('WebSocket readyState:', ws.readyState);
        setConnectionStatus('Error');
      };

      return ws;
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setConnectionStatus('Error');
      return null;
    }
  }, [url]);

  useEffect(() => {
    const ws = connect();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (ws) {
        ws.close(1000, 'Component unmounting');
      }
    };
  }, [connect]);

  const sendMessage = useCallback((message: any) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, [socket]);

  return {
    socket,
    lastMessage,
    connectionStatus,
    sendMessage
  };
};