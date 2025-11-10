# Setup Guide

## Backend Dependencies

The API server requires WebSocket support. Install the required packages in your PyCharm environment:

```bash
pip install 'uvicorn[standard]' websockets fastapi pydantic python-multipart
```

Or install from requirements.txt:

```bash
pip install -r src/requirements.txt
```

## Running the System

1. **Start the API Server** (in PyCharm or terminal):
   ```bash
   cd unified/m1/src
   python api_server.py
   ```
   
   The server should start on http://localhost:8080

2. **Start the Frontend**:
   ```bash
   cd unified/m1/client
   npm install  # if not already done
   npm run dev
   ```
   
   The frontend will be available at http://localhost:5173 or similar

## Troubleshooting

### WebSocket Connection Issues

If you see "No supported WebSocket library detected":
- Install uvicorn with standard extras: `pip install 'uvicorn[standard]'`
- Or install websockets manually: `pip install websockets`

### CORS Issues

If the frontend can't connect to the backend:
- Make sure the backend is running on port 8080
- Check that no firewall is blocking the connection
- Verify the API responds: `curl http://localhost:8080/`

### Frontend Build Issues

If TypeScript compilation fails:
- Make sure all dependencies are installed: `npm install`
- Try clearing the cache: `rm -rf node_modules package-lock.json && npm install`

## Testing the Connection

You can test the WebSocket connection manually:

```bash
# Test HTTP API
curl http://localhost:8080/state

# Test WebSocket (requires wscat: npm install -g wscat)
wscat -c ws://localhost:8080/ws
```