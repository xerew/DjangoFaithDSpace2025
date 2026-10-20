"""Minimal Ollama client: model capabilities and one structured completion."""
import time

import requests


class OllamaClient:
    def __init__(self, base_url, timeout=300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._vision = {}

    def has_vision(self, model):
        """Whether the model accepts images (asked once per model)."""
        if model not in self._vision:
            response = requests.post(f'{self.base_url}/api/show', json={'model': model}, timeout=60)
            response.raise_for_status()
            data = response.json()
            capabilities = data.get('capabilities')
            if capabilities is not None:
                self._vision[model] = 'vision' in capabilities
            else:  # older Ollama versions
                families = (data.get('details') or {}).get('families') or []
                self._vision[model] = 'clip' in families or 'projector_info' in data
        return self._vision[model]

    def generate(self, model, prompt, schema, images, temperature, seed, think):
        """Return (reply text, seconds taken)."""
        payload = {
            'model': model,
            'prompt': prompt,
            'format': schema,
            'stream': False,
            'options': {'temperature': temperature, 'seed': seed},
        }
        if images:
            payload['images'] = images
        if think is not None:
            payload['think'] = think
        started = time.monotonic()
        response = requests.post(f'{self.base_url}/api/generate', json=payload, timeout=self.timeout)
        response.raise_for_status()
        envelope = response.json()
        text = (envelope.get('response') or envelope.get('thinking') or '').strip()
        return text, time.monotonic() - started
