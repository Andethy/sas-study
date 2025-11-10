import React, { useEffect, useRef } from 'react';

interface LogEntry {
  id: number;
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'error';
}

interface StatusLogProps {
  logs: LogEntry[];
}

const StatusLog: React.FC<StatusLogProps> = ({ logs }) => {
  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs are added
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="control-group">
      <h3>Status Log</h3>
      <div ref={logRef} className="status-log">
        {logs.length === 0 ? (
          <div className="log-entry">No messages yet...</div>
        ) : (
          logs.map((log) => (
            <div key={log.id} className={`log-entry ${log.type}`}>
              <span style={{ opacity: 0.7 }}>[{log.timestamp}]</span> {log.message}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default StatusLog;