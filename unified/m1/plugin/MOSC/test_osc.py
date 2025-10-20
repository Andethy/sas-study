#!/usr/bin/env python3
"""
Simple OSC test script for MOSC plugin
Sends various OSC messages to test the plugin functionality
"""

import argparse
import time
from pythonosc import udp_client

def test_note_messages(client):
    """Test /note OSC messages with different formats"""
    print("Testing /note messages...")
    
    # Basic note with default duration (0.1s)
    print("  Sending: /note 36 0.8")
    client.send_message("/note", [36, 0.8])
    time.sleep(0.5)
    
    # Note with custom duration
    print("  Sending: /note 36 0.6 0.5")
    client.send_message("/note", [36, 0.6, 0.5])
    time.sleep(1.0)
    
    # Note with custom duration and channel
    print("  Sending: /note 36 0.9 0.3 2")
    client.send_message("/note", [36, 0.9, 0.3, 2])
    time.sleep(0.8)
    
    # Chord
    print("  Sending chord...")
    for note in [60, 64, 67]:
        client.send_message("/note", [note, 0.7, 1.0])
    time.sleep(1.5)

def test_cc_messages(client):
    """Test /cc OSC messages"""
    print("Testing /cc messages...")
    
    # Basic CC message
    print("  Sending: /cc 1 64 1")
    client.send_message("/cc", [1, 64, 1])
    time.sleep(0.2)
    
    # CC with offset
    print("  Sending: /cc 7 100 1 100.0")
    client.send_message("/cc", [7, 100, 1, 100.0])
    time.sleep(0.5)
    
    # Multiple CC messages
    print("  Sending CC sweep...")
    for value in range(0, 128, 16):
        client.send_message("/cc", [11, value, 1])
        time.sleep(0.1)

def test_invalid_messages(client):
    """Test invalid messages to check error handling"""
    print("Testing invalid messages...")
    
    # Invalid /note (missing parameters)
    print("  Sending invalid /note (missing velocity)")
    client.send_message("/note", [60])
    time.sleep(0.2)
    
    # Invalid /cc (missing parameters)
    print("  Sending invalid /cc (missing channel)")
    client.send_message("/cc", [1, 64])
    time.sleep(0.2)
    
    # Unknown message
    print("  Sending unknown message /unknown")
    client.send_message("/unknown", [1, 2, 3])
    time.sleep(0.2)

def main():
    parser = argparse.ArgumentParser(description='Test OSC messages for MOSC plugin')
    parser.add_argument('--host', default='127.0.0.1', help='OSC host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9001, help='OSC port (default: 9001)')
    parser.add_argument('--test', choices=['note', 'cc', 'invalid', 'all'], default='all',
                        help='Which tests to run (default: all)')
    
    args = parser.parse_args()
    
    print(f"Connecting to OSC server at {args.host}:{args.port}")
    client = udp_client.SimpleUDPClient(args.host, args.port)
    
    print("Starting OSC tests...")
    print("Make sure your MOSC plugin is loaded and listening!")
    print()
    
    if args.test in ['note', 'all']:
        test_note_messages(client)
        print()
    
    if args.test in ['cc', 'all']:
        test_cc_messages(client)
        print()
    
    if args.test in ['invalid', 'all']:
        test_invalid_messages(client)
        print()
    
    print("Tests completed!")

if __name__ == '__main__':
    main()