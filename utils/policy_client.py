import asyncio
import traceback
import typing as t
import json

import aiohttp
import json_numpy
import numpy as np
import requests

json_numpy.patch()


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
        assert self._async_session is not None

        url = get_url(self.host, self.port, "reset")

        async with self._async_session.post(url) as resp:
            try:
                result = await resp.json()
                if result.get("status") != "reset successful":
                    raise RuntimeError(
                        f"Server returned status {result.get('status')} after reset"
                    )
            except Exception as e:
                text = await resp.text()
                raise RuntimeError(
                    f"Server returned non-JSON response after reset: {text}; {e}"
                )

    def __call__(
        self,
        obs_dict: dict[str, t.Any],
        language_instruction: t.Optional[str] = None,
        history_dict: t.Optional[dict[str, t.Any]] = None,
    ) -> tuple[np.ndarray, str]:
        """
        Synchronous version of the call method.
        """
        assert "image_primary" in obs_dict.keys()
        assert isinstance(obs_dict["image_primary"], np.ndarray)
        assert len(obs_dict["image_primary"].shape) == 3, obs_dict[
            "image_primary"
        ].shape

        request_data = {
            "image": obs_dict["image_primary"],
            "instruction": language_instruction,
            "test": True,
        }
        if history_dict is not None:
            request_data["history"] = history_dict

        # Use the session for connection reuse
        url = get_url(self.host, self.port, "act")
        response = self._session.post(url, json=request_data)
        resp = response.json()

        if isinstance(resp, list):
            # VLA policies return a list of actions, repackage for consistency
            resp = {"action": resp, "vlm_response": ""}

        if type(resp["action"]) not in (np.ndarray, list):
            raise RuntimeError(
                "Policy server returned invalid action. It must return a numpy array or a list. Received: "
                + str(resp)
            )
        return resp["action"], resp["vlm_response"]

    async def async_init(self) -> "PolicyClient":
        """Initialize the async session if it doesn't exist"""
        if self._async_session is None:
            self._async_session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    force_close=True,
                    limit=64,
                    enable_cleanup_closed=True,
                )
            )
        return self

    async def async_call(
        self,
        obs_dict: dict[str, t.Any],
        language_instruction: t.Optional[str] = None,
        history_dict: t.Optional[dict[str, t.Any]] = None,
    ) -> tuple[np.ndarray, str]:
        """Async version of the call method"""
        assert "image_primary" in obs_dict.keys()
        assert isinstance(obs_dict["image_primary"], np.ndarray)
        assert len(obs_dict["image_primary"].shape) == 3, obs_dict[
            "image_primary"
        ].shape

        if self._async_session is None:
            await self.async_init()
        assert self._async_session is not None

        max_retries = 3
        retry_delay = 1.0

        # have to manually jsonify the history dict for aiohttp
        if history_dict is not None:
            for step in history_dict["steps"]:
                step["images"] = [im.tolist() for im in step["images"]]

        for attempt in range(max_retries):
            try:
                async with self._async_session.post(
                    get_url(self.host, self.port, "act"),
                    json={
                        "image": obs_dict["image_primary"].tolist(),
                        "instruction": language_instruction,
                        "history": history_dict,
                        "test": True,
                    },
                    timeout=60,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Server returned error {response.status}: {error_text}")
                        if attempt < max_retries - 1:
                            print(
                                f"Retrying in {retry_delay}s... (attempt {attempt + 1} / {max_retries})"
                            )
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            raise RuntimeError(
                                f"Server returned status {response.status} after {max_retries} attempts. Body: {error_text[:500]}"
                            )

                    # Try to parse JSON, handle potential errors
                    try:
                        # json_numpy.patch() ensures that np.ndarray might be parsed directly
                        original_json_response = await response.json()
                    except aiohttp.client_exceptions.ContentTypeError as e:
                        # This occurs if the response content-type is not JSON (e.g. HTML error page)
                        error_text = await response.text()
                        print(f"Server returned non-JSON content (status {response.status}). Body preview: {error_text[:500]}. Error: {e}")
                        # Propagate as a RuntimeError to be caught by the retry logic or caller
                        raise RuntimeError(f"Server returned non-JSON response: {error_text[:500]}") from e
                    except json.JSONDecodeError as e: # If content-type is JSON but body is malformed
                        error_text = await response.text()
                        print(f"Failed to decode JSON response (status {response.status}). Body preview: {error_text[:500]}. Error: {e}")
                        raise RuntimeError(f"Failed to decode JSON response: {error_text[:500]}") from e
                    except Exception as e: # Catch-all for other unexpected errors during .json() or .text()
                        try:
                            error_text = await response.text()
                            print(f"Unexpected error processing server response (status {response.status}). Body preview: {error_text[:500]}. Error: {e}")
                        except Exception as text_e:
                            print(f"Unexpected error processing server response (status {response.status}). Additionally, failed to get response text: {text_e}. Original error: {e}")
                        raise # Re-raise the original error

                    # Process the successfully parsed JSON response
                    current_response_data = original_json_response
                    if isinstance(original_json_response, list):
                        # As per synchronous client behavior, a list response is treated as the action.
                        current_response_data = {"action": original_json_response, "vlm_response": ""}

                    if not isinstance(current_response_data, dict):
                        # This will now be caught by the retry logic if it's a persistent server issue
                        raise RuntimeError(
                            f"Processed server response is not a dictionary. Received: {current_response_data}"
                        )

                    if "action" not in current_response_data:
                        # This will now be caught by the retry logic
                        raise RuntimeError(
                            f"Policy server response dictionary is missing 'action' key. Received: {current_response_data}"
                        )

                    action_data = current_response_data["action"]
                    vlm_response_str = current_response_data.get("vlm_response", "") # Default if missing

                    # Ensure action_data is either np.ndarray or list, as per original logic
                    if not isinstance(action_data, (np.ndarray, list)):
                        # This will now be caught by the retry logic
                        raise RuntimeError(
                            f"Policy server 'action' field has invalid type. Must be numpy array or list. Received type: {type(action_data)}, Value: {action_data}"
                        )
                    
                    # Convert action to numpy array for consistent return type tuple[np.ndarray, str]
                    action_np = np.array(action_data) if not isinstance(action_data, np.ndarray) else action_data.copy()
                    
                    return action_np, vlm_response_str

            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e: # Added RuntimeError to catch our new raises
                if attempt < max_retries - 1:
                    print(
                        f"Request failed with error: {e}; traceback: {traceback.format_exc()}. Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                    )
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
