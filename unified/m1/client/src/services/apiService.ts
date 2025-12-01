const API_BASE = import.meta.env.PROD 
  ? window.location.origin 
  : 'http://localhost:8080';

interface Tensions {
  zone1: number;
  zone2: number;
  zone3: number;
}

interface OrchestratorState {
  bpm: number;
  beats_per_bar: number;
  key_root: string;
  tensions: Tensions;
  current_bar: number;
}

class ApiService {
  async request<T = any>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE}${endpoint}`;
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    console.log('Making API request:', { url, config });

    try {
      const response = await fetch(url, config);
      console.log('API response status:', response.status, response.statusText);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('API error response:', errorData);
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('API response data:', result);
      return result;
    } catch (error) {
      console.error('API request failed:', error);
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Unable to connect to orchestrator API. Please check if the server is running.');
      }
      throw error;
    }
  }

  async getState(): Promise<OrchestratorState> {
    return this.request<OrchestratorState>('/state');
  }

  async updateTension(tensions: Tensions): Promise<{ status: string; tensions: Tensions }> {
    console.log('API Service: Sending tension update:', tensions);
    const result = await this.request('/tension', {
      method: 'POST',
      body: JSON.stringify(tensions),
    });
    console.log('API Service: Received response:', result);
    return result;
  }

  async updateBPM(bpm: number): Promise<{ status: string; bpm: number }> {
    return this.request('/bpm', {
      method: 'POST',
      body: JSON.stringify({ bpm }),
    });
  }

  async updateKey(key: string): Promise<{ status: string; key: string }> {
    return this.request('/key', {
      method: 'POST',
      body: JSON.stringify({ key }),
    });
  }

  async updateBeatsPerBar(beats_per_bar: number): Promise<{ status: string; beats_per_bar: number }> {
    return this.request('/beats_per_bar', {
      method: 'POST',
      body: JSON.stringify({ beats_per_bar }),
    });
  }

  async queueChord(chord_symbol: string, group_id: string = 'harm'): Promise<{ status: string; chord: string }> {
    return this.request('/chord', {
      method: 'POST',
      body: JSON.stringify({ chord_symbol, group_id }),
    });
  }

  async triggerFill(preset: string, beats: number): Promise<{ status: string; preset: string; beats: number }> {
    return this.request('/fill', {
      method: 'POST',
      body: JSON.stringify({ preset, beats }),
    });
  }

  // Timbre Interpolation Methods
  async uploadTimbreSample(file: File, sampleType: 'sample_a' | 'sample_b'): Promise<{ filename: string; file_id: string; status: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/timbre/upload/${sampleType}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  }

  async setTimbreMix(mixValue: number): Promise<{ status: string; mix_value: number; output_file?: string; audio_url?: string }> {
    return this.request('/timbre/mix', {
      method: 'POST',
      body: JSON.stringify({ mix_value: mixValue }),
    });
  }

  async getTimbreStatus(): Promise<{ sample_a: string | null; sample_b: string | null; current_mix: number; ready: boolean; realtime_status?: any }> {
    return this.request('/timbre/status');
  }

  async startRealtimePlayback(): Promise<{ status: string; message: string }> {
    return this.request('/timbre/playback/start', {
      method: 'POST',
    });
  }

  async stopRealtimePlayback(): Promise<{ status: string; message: string }> {
    return this.request('/timbre/playback/stop', {
      method: 'POST',
    });
  }

  async deleteTimbreSample(sampleType: 'sample_a' | 'sample_b'): Promise<{ status: string; sample_type: string }> {
    return this.request(`/timbre/${sampleType}`, {
      method: 'DELETE',
    });
  }
}

export const apiService = new ApiService();