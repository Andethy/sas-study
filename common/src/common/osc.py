from typing import Dict, Any, Iterable, Tuple, Union
from pythonosc import udp_client

Scalar = Union[int, float, str, bytes, bool]
Arg = Union[Scalar, Iterable[Scalar]]

def _coerce_args(val: Any, extra: Tuple[Any, ...]) -> list:
    """
    Normalize OSC args:
    - If val is iterable (not str/bytes), flatten one level to a list.
    - Append any *extra args.
    - Coerce numpy scalar types to Python scalars if present.
    """
    def _to_py(x):
        # Avoid importing numpy; duck-convert common attrs
        if hasattr(x, "item") and callable(getattr(x, "item")):
            try:
                return x.item()
            except Exception:
                return x
        return x

    args: list = []
    if isinstance(val, (list, tuple)):
        args.extend(_to_py(x) for x in val)
    elif isinstance(val, (str, bytes)):
        args.append(val)  # strings/bytes are scalar payloads
    else:
        args.append(_to_py(val))
    if extra:
        args.extend(_to_py(x) for x in extra)
    return args

class OSCManager:
    ports: Dict[int, udp_client.SimpleUDPClient]

    def __init__(self, address: str, *ports: int, base: int = -1, k: int = 0, **kwargs: Any):
        self.address = address
        self.ports: Dict[int, udp_client.SimpleUDPClient] = {}
        self.base = base
        self.k = k
        self._i = 0

        # Case A: explicit port list
        for port in ports:
            print(f'Setup client on port {port} (de)')
            self.ports[port] = udp_client.SimpleUDPClient(address, port)

        # Case B: contiguous range starting at base, length k
        for i in range(k):
            print(f'Setup client on port {base + i} (bi)')
            self.ports[base + i] = udp_client.SimpleUDPClient(address, base + i)

        # Case C: custom mapping via kwargs {port: client}
        for port, client in kwargs.items():
            print(f'Setup client on port {port} (dc)')
            # noinspection PyTypeChecker
            self.ports[int(port)] = client

    # BACKWARD-COMPAT: still callable like mgr(port, addr, val[, *more])
    def __call__(self, port: int, addr: str, val: Arg, *args: Any, **kwargs: Any) -> None:
        # print(f'Sending msg on port {port} to {addr} set to {val}') #!@
        client = self.ports[port]
        payload = _coerce_args(val, args)
        client.send_message(addr, payload if len(payload) > 1 else payload[0])

    def send(self, port: int, addr: str, *args: Any) -> None:
        """Explicit send with arbitrary arg list."""
        client = self.ports[port]
        payload = _coerce_args(list(args), ())
        # print(f"Sending msg on port {port} to {addr} args={payload}")
        client.send_message(addr, payload if len(payload) > 1 else payload[0])

    def broadcast(self, addr: str, *args: Any) -> None:
        """Send the same message to all configured ports."""
        payload = _coerce_args(list(args), ())
        for port, client in self.ports.items():
            print(f"Broadcast to port {port} -> {addr} {payload}")
            client.send_message(addr, payload if len(payload) > 1 else payload[0])

    def __getitem__(self, item: int) -> udp_client.SimpleUDPClient:
        return self.ports[item]

    # BACKWARD-COMPAT: allow mgr[port, addr] = value  (value can be scalar or list)
    def __setitem__(self, key, value) -> None:
        port, addr = key
        client = self.ports[port]
        payload = _coerce_args(value, ())
        client.send_message(addr, payload if len(payload) > 1 else payload[0])

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self._i >= self.k:
            self._i = 0
            raise StopIteration
        self._i += 1
        return self.base + (self._i - 1)

    def reconnect(self) -> None:
        for port in list(self.ports):
            self.ports[port] = udp_client.SimpleUDPClient(self.address, port)