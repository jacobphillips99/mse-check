import numpy as np
import requests
import aiohttp
import asyncio
import time
import typing as t


def get_url(
    host: str, port: int, endpoint: str | None = None, protocol: str = "http://"
):
    """
    Get the URL for a given host and port; if port is negative, skip it.
    Cleans the host and endpoint strings
    """
    # Remove http:// or https:// from host if present, 
    host_str = host.replace("http://", "").replace("https://", "")
    port_str = f":{port}" if int(port) >= 0 else ""
    endpoint_str = f"/{endpoint.lstrip('/')}" if endpoint else ""
    return f"{protocol}{host_str}{port_str}{endpoint_str}"


class PolicyClient:
    """
    A simple client to query actions from the policy server
    """
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._session = requests.Session()
        self._async_session = None

    async def server_reset(self) -> None:
        if self._async_session is None:
            await self.async_init()

        url = get_url(self.host, self.port, "reset")
        
        async with self._async_session.post(url) as resp:
            try:
                result = await resp.json()
                if result.get("status") != "reset successful":
                    raise RuntimeError(f"Server returned status {result.get('status')} after reset")
            except:
                text = await resp.text()
                raise RuntimeError(f"Server returned non-JSON response after reset: {text}")

    def __call__(self, obs_dict: dict[str, t.Any], language_instruction: t.Optional[str] = None) -> np.ndarray:
        assert "image_primary" in obs_dict.keys()
        assert isinstance(obs_dict["image_primary"], np.ndarray)
        assert len(obs_dict["image_primary"].shape) == 3, obs_dict["image_primary"].shape

        request_data = {
            "image": obs_dict["image_primary"],
            "instruction": language_instruction,
        }
        # Use the session for connection reuse
        url = get_url(self.host, self.port, "act")
        response = self._session.post(url, json=request_data)
        action = response.json()
       
        # the original action is not modifiable, cannot clip boundaries after the fact for example
        if type(action) not in (np.ndarray, list):
            raise RuntimeError(
                "Policy server returned invalid action. It must return a numpy array or a list. Received: "
                + str(action)
            )
        return action.copy()
    
    async def async_init(self) -> "PolicyClient":
        """Initialize the async session if it doesn't exist"""
        if self._async_session is None:
            self._async_session = aiohttp.ClientSession()
        return self
    
    async def async_call(self, obs_dict: dict[str, t.Any], language_instruction: t.Optional[str] = None) -> np.ndarray:
        """Async version of the call method"""
        assert "image_primary" in obs_dict.keys()
        assert isinstance(obs_dict["image_primary"], np.ndarray)
        assert len(obs_dict["image_primary"].shape) == 3, obs_dict["image_primary"].shape
        
        if self._async_session is None:
            await self.async_init()
        
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                async with self._async_session.post(
                    get_url(self.host, self.port, "act"),
                    json={
                        "image": obs_dict["image_primary"],
                        "instruction": language_instruction,
                    },
                    timeout=30
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Server returned error {response.status}: {error_text}")
                        if attempt < max_retries - 1:
                            print(f"Retrying in {retry_delay}s... (attempt {attempt+1}/{max_retries})")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            raise RuntimeError(f"Server returned status {response.status} after {max_retries} attempts")
                    
                    # Try to parse JSON, handle potential errors
                    try:
                        action = await response.json()
                    except aiohttp.client_exceptions.ContentTypeError:
                        # Fallback to text if JSON parsing fails
                        text_response = await response.text()
                        print(f"Warning: Failed to parse JSON response: {text_response[:100]}...")
                        raise RuntimeError(f"Server returned invalid response format (not JSON)")
                                        
                    if type(action) not in (np.ndarray, list):
                        raise RuntimeError(
                            "Policy server returned invalid action. It must return a numpy array or a list. Received: "
                            + str(action)
                        )
                    return action.copy()
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    print(f"Request failed with error: {e}. Retrying in {retry_delay}s... (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Request failed after {max_retries} attempts: {e}")
                    raise
    
    def close(self) -> None:
        """Explicitly close connections"""
        if hasattr(self, "_session"):
            self._session.close()
        
        # Fix the async session closure
        if hasattr(self, "_async_session") and self._async_session is not None:
            # Don't use asyncio.run() here, just create a task to close it
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._async_session.close())
                else:
                    # If loop is not running, we can run it briefly just to close
                    loop.run_until_complete(self._async_session.close())
            except RuntimeError:
                # If we can't get the loop or the loop is closed, just set to None
                pass
            self._async_session = None

    def __del__(self) -> None:
        self.close()