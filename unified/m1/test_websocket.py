#!/usr/bin/env python3
"""
Simple WebSocket client to test the orchestrator API WebSocket endpoint
"""
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8080/ws"
    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("Connected successfully!")
            
            # Wait for initial state message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Received initial message: {data}")
            
            # Send a ping
            await websocket.send("ping")
            
            # Wait for response
            response = await websocket.recv()
            print(f"Received response: {response}")
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())